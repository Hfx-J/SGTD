#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <ikd-Tree/ikd_Tree.h>
#include <yaml-cpp/yaml.h>
#include <unordered_map>
// #include <boost/filesystem.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <chrono>
#include <ctime>
#include "cluster_manager.hpp"
#include "ndt_3d.h"
#include "matplotlibcpp.h"
#include <nlohmann/json.hpp>
#include "lapjav.hpp"
#include "linear_sum_assignment.hpp"
#include "ssc.h"
#ifdef USE_VGICP_CUDA
#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif
#include <fast_gicp/gicp/fast_gicp.hpp>

/**
 * @brief      Calculates the Root Mean Square Error (RMSE) from a vector of errors.
 * @param[in]  x: A vector of error values.
 * 
 * @return     The RMSE, which is the square root of the average squared errors.
 * 
 * This function computes the Root Mean Square Error (RMSE) by squaring each error, 
 * averaging them, and then taking the square root of the result. RMSE is commonly used
 * to quantify the accuracy of a model's predictions, especially in regression tasks.
 */
double calculateRMSE(const std::vector<double>& x) {
    double sum_square_error = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum_square_error += x[i] * x[i];  // Accumulate squared errors
    }
    double mean_square_error = sum_square_error / x.size();  // Calculate mean squared error
    double rmse = std::sqrt(mean_square_error);  // Take square root to compute RMSE
    return rmse;  // Return the computed RMSE
}

/**
 * @brief      Calculates the angular difference between two rotation matrices.
 * @param[in]  T1: The first transformation matrix.
 * @param[in]  T2: The second transformation matrix.
 * 
 * @return     The angular difference in degrees between the two rotations.
 * 
 * This function computes the angular difference between two transformations based on 
 * their rotation matrices. The angle is calculated using the trace of the relative rotation matrix
 * and is then converted from radians to degrees.
 */
double calculateAngleDifference(const Eigen::Matrix4f& T1, const Eigen::Matrix4f& T2) {
    Eigen::Matrix3f R1 = T1.block<3, 3>(0, 0);  // Extract the rotation matrix from T1
    Eigen::Matrix3f R2 = T2.block<3, 3>(0, 0);  // Extract the rotation matrix from T2
    
    Eigen::Matrix3f R_relative = R1 * R2.transpose();  // Compute the relative rotation matrix
    
    double trace = R_relative.trace();  // Compute the trace of the relative rotation matrix
    
    // Clamp the trace to the valid range [-1, 1] to avoid numerical errors when computing acos
    trace = std::max(-1.0, std::min(3.0, trace));  
    double angle_rad = std::acos((trace - 1.0) / 2.0);  // Compute the angle in radians
    
    double angle_deg = angle_rad * 180.0 / M_PI;  // Convert the angle from radians to degrees
    
    return angle_deg;  // Return the angular difference in degrees
}

/**
 * @brief      Computes the relative translation and rotation errors between two transformations.
 * @param[in]  gt: The ground truth transformation matrix.
 * @param[in]  lo: The localization transformation matrix.
 * @param[out] t_e: The computed translation error.
 * @param[out] r_e: The computed rotation error.
 * 
 * This function computes the relative pose error (RPE) by comparing the ground truth 
 * transformation with the localization transformation. It calculates both translation and
 * rotation errors, with the rotation error computed as the angular difference between the
 * relative rotation matrices.
 */
void compute_adj_rpe(Eigen::Matrix4f& gt,
                     Eigen::Matrix4f& lo,
                     double& t_e,
                     double& r_e) {
    Eigen::Matrix4f delta_T = lo.inverse() * gt;  // Compute the relative transformation

    t_e = delta_T.topRightCorner(3, 1).norm();  // Calculate the translation error (Euclidean norm)

    // Compute the rotation error as the angular difference between the rotation matrices
    r_e = std::abs(std::acos(
                  fmin(fmax((delta_T.block<3, 3>(0, 0).trace() - 1) / 2, -1.0),
                       1.0))) /
          M_PI * 180;  // Convert the angle from radians to degrees
}


/**
 * @brief      Computes the centroid (average point) of a set of points.
 * @param[in]  points: A collection of 3D points.
 * 
 * @return     The centroid of the points as an Eigen::Vector3f.
 * 
 * This function calculates the centroid of a given set of points. The centroid is the
 * arithmetic mean position of all the points in the set, and it is commonly used in
 * point cloud processing or for finding the center of a cluster of points.
 */
Eigen::Vector3f compute_centroid(const Points &points) {
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();  // Initialize centroid to the zero vector
    for (const auto &point : points) {
        centroid += point;  // Sum all the points
    }
    return centroid / points.size();  // Divide by the number of points to get the average (centroid)
}


/**
 * @brief      Extracts corresponding points from two sets of points based on a given index pair.
 * @param[in]  points_i: The first set of points (source points).
 * @param[in]  points_j: The second set of points (destination points).
 * @param[in]  pair: A vector of indices where each element corresponds to the index of a matching point in `points_j` for the point in `points_i`.
 * @param[out] src_points: The extracted source points from `points_i` corresponding to the valid indices.
 * @param[out] dst_points: The extracted destination points from `points_j` corresponding to the valid indices.
 * 
 * This function extracts corresponding points between two point sets (`points_i` and `points_j`) based on a pair of indices provided.
 * If an index in the `pair` vector is not `-1`, the corresponding points from both sets are added to `src_points` and `dst_points`, respectively.
 */
void extract_corresponding_points(const Points &points_i, const Points &points_j, const IntVector &pair, Points &src_points, Points &dst_points) {
    for (int idx = 0; idx < pair.size(); ++idx) {  // Loop through each index in the pair vector
        if (pair[idx] != -1) {  // Check if the index in the pair vector is valid (not -1)
            src_points.push_back(points_i[idx]);  // Add the source point corresponding to the current index
            dst_points.push_back(points_j[pair[idx]]);  // Add the destination point corresponding to the matching index
        }
    }
}


/**
 * @brief      Computes the transformation (rotation and translation) between two sets of 3D points.
 * @param[in]  points_i: The first set of 3D points (source points).
 * @param[in]  points_j: The second set of 3D points (destination points).
 * @param[in]  pair: A vector of indices where each element corresponds to a matching point index in `points_j` for the point in `points_i`.
 * @param[out] R: The 3x3 rotation matrix representing the transformation.
 * @param[out] t: The 3D translation vector representing the transformation.
 * 
 * This function computes the best rigid body transformation (rotation and translation) that aligns two sets of 3D points (`points_i` and `points_j`),
 * based on the corresponding point pairs provided in the `pair` vector. The transformation is computed using the centroid method and singular value decomposition (SVD).
 */
void compute_transformation(const Points &points_i, const Points &points_j, const IntVector &pair, Eigen::Matrix3f &R, Eigen::Vector3f &t) {
    Points src_points, dst_points;
    extract_corresponding_points(points_i, points_j, pair, src_points, dst_points);  // Extract corresponding points based on the pair

    // Compute centroids of the source and destination points
    Eigen::Vector3f centroid_src = compute_centroid(src_points);  // Centroid of the source points
    Eigen::Vector3f centroid_dst = compute_centroid(dst_points);  // Centroid of the destination points

    // Subtract centroids to center the points at the origin (for translation invariance)
    for (auto &point : src_points) point -= centroid_src;  // Shift source points by subtracting centroid
    for (auto &point : dst_points) point -= centroid_dst;  // Shift destination points by subtracting centroid

    // Compute the cross-covariance matrix H between the source and destination points
    Eigen::Matrix3f H = Eigen::Matrix3f::Zero();  // Initialize H to a zero matrix
    for (int i = 0; i < src_points.size(); ++i) {
        H += src_points[i] * dst_points[i].transpose();  // Accumulate the outer product of the centered points
    }

    // Perform Singular Value Decomposition (SVD) on the covariance matrix H
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);  // Compute SVD
    Eigen::Matrix3f U = svd.matrixU();  // Left singular vectors
    Eigen::Matrix3f Vt = svd.matrixV().transpose();  // Right singular vectors (transposed)

    // Compute the rotation matrix R from the SVD results
    R = Vt.transpose() * U.transpose();  // Rotation matrix obtained from SVD

    // Ensure that the rotation matrix has a positive determinant (to avoid reflection)
    if (R.determinant() < 0) {
        Vt.row(2) *= -1;  // Flip the sign of the third row to correct the determinant
        R = Vt.transpose() * U.transpose();  // Recompute the rotation matrix with corrected Vt
    }

    // Compute the translation vector t based on the centroids and rotation matrix
    t = centroid_dst - R * centroid_src;  // Translation is the difference between destination and rotated source centroids
}

/**
 * @brief      Evaluates the quality of the transformation between two sets of 3D points.
 * @param[in]  points_i: The first set of 3D points (source points).
 * @param[in]  points_j: The second set of 3D points (destination points).
 * @param[in]  pair: A vector of indices where each element corresponds to a matching point index in `points_j` for the point in `points_i`.
 * @param[in]  R: The 3x3 rotation matrix representing the transformation.
 * @param[in]  t: The 3D translation vector representing the transformation.
 * 
 * @return     The mean squared error (MSE) between the transformed source points and the destination points.
 * 
 * This function computes the Mean Squared Error (MSE) between the transformed source points (using the given rotation `R` and translation `t`)
 * and the destination points. It is used to evaluate how well the transformation aligns the source points with the destination points.
 */
double evaluate_transformation(const Points &points_i, const Points &points_j, const IntVector &pair, const Eigen::Matrix3f &R, const Eigen::Vector3f &t) {
    Points src_points, dst_points;
    extract_corresponding_points(points_i, points_j, pair, src_points, dst_points);  // Extract corresponding points based on the pair

    // Initialize mean squared error (MSE) to zero
    double mse = 0.0;

    // Calculate the squared distance between the transformed source points and the destination points
    for (int i = 0; i < src_points.size(); ++i) {
        Eigen::Vector3f transformed_point = R * src_points[i] + t;  // Apply the transformation (rotation + translation) to the source point
        mse += (transformed_point - dst_points[i]).squaredNorm();  // Compute the squared Euclidean distance and accumulate
    }

    // Normalize the MSE by the number of points
    mse /= src_points.size();  // Compute the mean squared error
    return mse;  // Return the MSE value
}


/**
 * @brief      Converts a PCL point cloud to a ROS PointCloud2 message.
 * @param[in]  cloud: The input PCL point cloud of type `T` (could be `pcl::PointXYZ`, `pcl::PointXYZRGB`, etc.).
 * @param[in]  frame_id: The coordinate frame ID to be set in the ROS message header. Default is "map".
 * 
 * @return     A ROS `PointCloud2` message containing the point cloud data.
 * 
 * This function converts a given PCL point cloud to a ROS `PointCloud2` message and sets the frame ID
 * in the header to the specified `frame_id`. If no frame ID is provided, it defaults to "map".
 */
template<typename T>
inline sensor_msgs::PointCloud2 cloud2msg(pcl::PointCloud<T> cloud, std::string frame_id = "map") {
    sensor_msgs::PointCloud2 cloud_ROS;  // Declare a ROS PointCloud2 message

    pcl::toROSMsg(cloud, cloud_ROS);  // Convert PCL point cloud to ROS message

    cloud_ROS.header.frame_id = frame_id;  // Set the frame ID for the ROS message header
    return cloud_ROS;  // Return the ROS PointCloud2 message
}


/**
 * @brief      Converts a vector of calibration matrix data to a 4x4 transformation matrix.
 * @param[in]  matrix_data: A vector containing the calibration matrix data. The vector should contain exactly 12 elements.
 * @param[out] calib_mat: The output 4x4 transformation matrix to store the calibration data.
 * 
 * @return     A boolean value indicating whether the conversion was successful. Returns `true` if the matrix data 
 *             contains exactly 12 elements and the conversion is done, `false` otherwise.
 * 
 * This function takes a vector of 12 float elements representing a 3x4 matrix (row-major format) and
 * converts it into a 4x4 transformation matrix (`Eigen::Matrix4f`). The function assumes that the last row 
 * of the calibration matrix is [0, 0, 0, 1] and sets the matrix identity for that row.
 */
bool vec2calib(const std::vector<float> matrix_data, Eigen::Matrix4f &calib_mat) {
    // Check if the input vector has exactly 12 elements
    if(matrix_data.size() != 12)
    {
        std::cerr << "[Read Calib Error!] Calibration matrix must have 12 elements." << std::endl;  // Error message if the vector size is incorrect
        return false;  // Return false if the size is not correct
    }

    calib_mat.setIdentity();  // Initialize the calibration matrix as identity matrix

    int index = 0;
    // Fill the 3x4 part of the calibration matrix (the first 3 rows, 4 columns)
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 4; ++j)
        {
            calib_mat(i, j) = matrix_data[index++];  // Assign values from the input vector to the matrix
        }
    }

    // The last row (index 3) is set as [0, 0, 0, 1] by default (already done by setIdentity())
    return true;  // Return true indicating successful conversion
}


/**
 * @brief      Colors a point cloud by assigning the same RGB color to all points.
 * @param[in]  cloud_in: A pointer to the input point cloud containing points of type `pcl::PointXYZ` (x, y, z).
 * @param[in]  color: A vector of 3 integers representing the RGB color values (r, g, b) for the points.
 * @param[out] cloud_out: A pointer to the output point cloud where the points will be stored with the assigned RGB colors.
 * 
 * This function takes an input point cloud (`cloud_in`) of type `pcl::PointXYZ`, where each point contains the
 * 3D coordinates (x, y, z), and assigns the same RGB color to all the points. The color is specified by the 
 * input `color` vector containing the RGB values. The output cloud (`cloud_out`) will be a `pcl::PointCloud<pcl::PointXYZRGB>`,
 * where each point contains both the 3D coordinates and the color information.
 */
void color_point_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
                       std::vector<int> color,
                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out) {
    // Extract RGB color values from the input vector
    int r = color[0];  // Red channel
    int g = color[1];  // Green channel
    int b = color[2];  // Blue channel

    pcl::PointXYZRGB temp_pt;  // Temporary point to store colored points

    // Iterate through each point in the input point cloud
    for (int i = 0; i < cloud_in->points.size(); ++i) {
        // Copy the coordinates from the input cloud to the temporary point
        temp_pt.x = cloud_in->points[i].x;
        temp_pt.y = cloud_in->points[i].y;
        temp_pt.z = cloud_in->points[i].z;

        // Assign the specified RGB color to the point
        temp_pt.r = r;
        temp_pt.g = g;
        temp_pt.b = b;

        // Add the colored point to the output cloud
        cloud_out->points.push_back(temp_pt);
    }
}

/**
 * @brief      Saves a vector of values to a file, one value per line.
 * @param[in]  vec: The vector containing the values to be saved.
 * @param[in]  filename: The path of the file where the vector values will be saved.
 * 
 * This function saves the elements of the input vector `vec` to a text file. Each element of the vector will be
 * written to a new line in the file specified by `filename`.
 */
template<typename T>
void saveVectorToFile(const std::vector<T>& vec, const std::string& filename) {
    // Open the file in write mode
    std::ofstream outFile(filename);
    
    // Check if the file opened successfully
    if (!outFile) {
        std::cerr << "outFile: " << filename << std::endl;  // Print error message if file opening failed
        return;
    }

    // Write each element of the vector to the file, one per line
    for (const T& value : vec) {
        outFile << value << std::endl;
    }

    // Close the file after writing
    outFile.close();

    // Log the successful file output
    std::cout << "outFile: " << filename << std::endl;
}

/**
 * @brief      Reads filenames from a folder and stores them in a vector.
 * @param[in]  folderName: The folder path where the files are located.
 * @param[in]  file_list_extenstion: The extension of the file containing the list of filenames (e.g., ".txt").
 * @param[in]  extension: The file extension to filter (e.g., ".jpg").
 * @param[out] fileNames: A vector where the file paths will be stored.
 * @param[in]  frame_begin: The starting frame number (default is 0).
 * @param[in]  frame_end: The ending frame number (default is 99999).
 * @param[in]  frame_step: The step size for frame numbers (default is 1).
 * 
 * This function reads the filenames from a file (specified by `folderName` and `file_list_extenstion`) and stores
 * the full paths of the files with the specified extension (`extension`) in the `fileNames` vector. The frames 
 * are filtered based on the `frame_begin`, `frame_end`, and `frame_step` parameters.
 * 
 * @return     `true` if the filenames were read successfully, `false` otherwise.
 */
bool batch_read_filenames_in_folder(const std::string &folderName,
                                    const std::string &file_list_extenstion,
                                    const std::string &extension,
                                    std::vector<std::string> &fileNames,
                                    int frame_begin = 0,
                                    int frame_end = 99999,
                                    int frame_step = 1) {
    // Construct the filename of the file containing the list of filenames
    std::string filename_list = folderName + file_list_extenstion;

    // Open the file containing the list of filenames
    std::ifstream name_list_file(filename_list.c_str(), std::ios::in);
    
    // Check if the file was opened successfully
    if (!name_list_file.is_open()) {
        // Log a message if the file cannot be opened
        return 0;
    }

    int frame_count = 0;  // Initialize frame count

    // Read the filenames from the file
    while (name_list_file.peek() != EOF) {
        std::string cur_file;
        name_list_file >> cur_file;  // Read the current filename

        // If the filename is not empty, proceed
        if (!cur_file.empty()) {
            // Check if the current frame is within the specified range
            if (frame_count >= frame_begin && frame_count <= frame_end &&
                ((frame_count - frame_begin) % frame_step == 0)) {
                cur_file = folderName + "/" + cur_file;  // Append the folder path to the filename
                fileNames.push_back(cur_file);  // Add the full file path to the output vector
            }
            frame_count++;  // Increment the frame count
        }
        std::cout<<"frame_count:"<<frame_count<<std::endl;
    }
    name_list_file.close();  // Close the file after reading

    return 1;  // Return success
}


/**
 * @brief      Converts an Eigen point cloud to a PCL PointCloud.
 * @param[in]  eigenPoints: A vector of Eigen::Vector3f points to be converted.
 * 
 * @return     A shared pointer to a pcl::PointCloud<pcl::PointXYZ> that contains the converted points.
 * 
 * This function takes a vector of Eigen::Vector3f points (`eigenPoints`) and converts it to a PCL PointCloud.
 * Each point from the Eigen vector is used to create a corresponding pcl::PointXYZ point which is then added 
 * to a new pcl::PointCloud. The resulting point cloud is returned as a shared pointer.
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr convertToPCLPointCloud(const Points& eigenPoints) {
    // Create a new pcl::PointCloud of type pcl::PointXYZ
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // Iterate over the Eigen::Vector3f points
    for (const auto& point : eigenPoints) {
        // Create a pcl::PointXYZ point and set its x, y, and z coordinates from the Eigen point
        pcl::PointXYZ pcl_point;
        pcl_point.x = point.x();
        pcl_point.y = point.y();
        pcl_point.z = point.z();

        // Add the pcl::PointXYZ point to the pcl::PointCloud
        cloud->points.push_back(pcl_point);
    }

    // Set the width and height of the point cloud
    cloud->width = static_cast<uint32_t>(cloud->points.size());  // Number of points
    cloud->height = 1;  // Assuming it's an unordered point cloud (single row)
    cloud->is_dense = true;  // The cloud is dense if there are no invalid points

    // Return the pcl::PointCloud pointer
    return cloud;
}


/**
 * @brief      Applies a color mapping based on a given label.
 * @param[in]  label: The label representing a specific class or object type.
 * @param[out] r: The red component of the RGB color.
 * @param[out] g: The green component of the RGB color.
 * @param[out] b: The blue component of the RGB color.
 * 
 * This function assigns RGB color values to the given label. The label represents different 
 * object types in a scene, such as vehicles, buildings, and terrain. The RGB values are 
 * used for visualizing these labels in a point cloud or other graphical representation.
 */
void apply_color_mapping_spvnas(int label, int &r, int &g, int &b) {
    // Switch case based on the input label value
    switch (label) {
    case 0: // car, bicycle, motorcycle, truck, other-vehicle, person, bicyclist, motorcyclist, road
        r = 100;
        g = 150;
        b = 245;
        break;
    case 3: // sidewalk
        r = 75;
        g = 0;
        b = 75;
        break;
    case 4: // other-ground
        r = 175;
        g = 0;
        b = 75;
        break;
    case 5: // building
        r = 255;
        g = 200;
        b = 0;
        break;
    case 6: // fence
        r = 255;
        g = 120;
        b = 50;
        break;
    case 7: // vegetation
        r = 0;
        g = 175;
        b = 0;
        break;
    case 8: // trunk
        r = 135;
        g = 60;
        b = 0;
        break;
    case 9: // terrain
        r = 150;
        g = 240;
        b = 80;
        break;
    case 10: // pole
        r = 255;
        g = 240;
        b = 150;
        break;
    case 11: // traffic-sign
        r = 255;
        g = 0;
        b = 0;
        break;
    default: // default case for moving objects or unknown labels
        r = 0;
        g = 0;
        b = 0;
    }
}

/**
 * @brief      Creates a marker for visualizing correspondences between two point clouds.
 * @param[in]  src_matched: The source point cloud containing matched points.
 * @param[in]  tgt_matched: The target point cloud containing matched points.
 * @param[out] marker: The ROS visualization marker that will hold the corresponding points.
 * @param[in]  thickness: The thickness of the lines connecting corresponding points. Default is 0.1.
 * @param[in]  rgb_color: The RGB color of the lines (default is black {0.0, 0.0, 0.0}).
 * @param[in]  id: The unique ID of the marker (default is 0).
 * 
 * This function generates a `visualization_msgs::Marker` of type `LINE_LIST` to represent
 * correspondences between points in two point clouds (source and target). Each pair of 
 * corresponding points is connected with a line, and the lines are visualized with the specified color 
 * and thickness. The marker is designed for use with ROS visualization tools such as `rviz`.
 */
void setCorrespondenceMarker(const pcl::PointCloud<pcl::PointXYZ> &src_matched,
                             const pcl::PointCloud<pcl::PointXYZ> &tgt_matched,
                             visualization_msgs::Marker &marker,
                             float thickness = 0.1,
                             std::vector<float> rgb_color = {0.0, 0.0, 0.0},
                             int id = 0) {
    // Clear existing points in the marker
    if (!marker.points.empty()) marker.points.clear();
    
    // Set the header information for the marker
    marker.header.frame_id = "map";  // Set the reference frame for the marker
    marker.header.stamp = ros::Time();  // Set the timestamp to the current time
    marker.ns = "my_namespace";  // Namespace for the marker
    marker.id = id;  // Unique ID to avoid overlap with other markers
    marker.type = visualization_msgs::Marker::LINE_LIST;  // Set the marker type as LINE_LIST
    marker.action = visualization_msgs::Marker::ADD;  // Add the marker to the visualization

    // Set the scale and color properties of the lines
    marker.scale.x = thickness;  // Set the thickness of the lines
    marker.color.r = rgb_color[0];  // Red component of the color
    marker.color.g = rgb_color[1];  // Green component of the color
    marker.color.b = rgb_color[2];  // Blue component of the color
    marker.color.a = 1.0;  // Set alpha to 1 for full opacity

    // Define points for the marker (source and target correspondences)
    geometry_msgs::Point srcP;
    geometry_msgs::Point tgtP;
    
    // Ensure that both point clouds have the same number of points
    assert(src_matched.size() == tgt_matched.size());
    
    // Iterate over each pair of corresponding points and add them to the marker
    for (int idx = 0; idx < src_matched.size(); ++idx) {
        pcl::PointXYZ sP = src_matched[idx];  // Get the source point
        pcl::PointXYZ sT = tgt_matched[idx];  // Get the target point

        // Set the coordinates of the source point
        srcP.x = sP.x;
        srcP.y = sP.y;
        srcP.z = sP.z;
        
        // Set the coordinates of the target point
        tgtP.x = sT.x;
        tgtP.y = sT.y;
        tgtP.z = sT.z;

        // Add the source and target points to the marker (forming a line)
        marker.points.emplace_back(srcP);
        marker.points.emplace_back(tgtP);
    }
}


/**
 * @brief      Color point cloud according to per point semantic labels.
 * @param[in]  semantic_cloud: input semantic cloud ptr (with label)
 * @param[in]  colored_cloud:  colored cloud ptr
 */
void color_pc(const Points& semantic_cloud,
              const vector<int>& label,
              pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud) {
  int r, g, b;
  uint16_t temp_label;
  pcl::PointXYZRGB temp_pt;
  for (int i = 0; i < semantic_cloud.size(); ++i) {
    temp_pt.x = semantic_cloud[i](0);
    temp_pt.y = semantic_cloud[i](1);
    temp_pt.z = semantic_cloud[i](2);
    temp_label = label[i];
    apply_color_mapping_spvnas((int)temp_label, r, g, b);
    temp_pt.r = r;
    temp_pt.g = g;
    temp_pt.b = b;
    colored_cloud->points.push_back(temp_pt);
  }
}

/**
 * @brief      Check if a substring exists in a string.
 * @param[in]  str: The string to check.
 * @param[in]  target: The substring to find.
 * 
 * @return     True if the substring exists in the string, false otherwise.
 */
bool contains(const std::string& str, const std::string& target) {
    return str.find(target) != std::string::npos;
}

/**
 * @brief      Color point cloud according to per point semantic labels.
 * @param[in]  semantic_cloud: input semantic cloud ptr (with label)
 * @param[in]  colored_cloud:  colored cloud ptr
 */
void Graph2CloudL(const Points& semantic_cloud,
              const vector<int>& label,
              pcl::PointCloud<pcl::PointXYZL>::Ptr colored_cloud) {
  int r, g, b;
  uint16_t temp_label;
  pcl::PointXYZL temp_pt;
  for (int i = 0; i < semantic_cloud.size(); ++i) {
    temp_pt.x = semantic_cloud[i](0);
    temp_pt.y = semantic_cloud[i](1);
    temp_pt.z = semantic_cloud[i](2);
    temp_pt.label = label[i];
    colored_cloud->points.push_back(temp_pt);
  }
}

void saveBinFile(const std::string& file_path, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud ,const std::string& final_p) {

    size_t last_slash_idx = file_path.find_last_of("/\\");
    std::string raw_name = file_path.substr(last_slash_idx + 1);
    size_t dot_idx = raw_name.find_last_of('.');
    std::string base_name = raw_name.substr(0, dot_idx);
    std::string file_name;

    std::string final_path = final_p + base_name + ".bin";

    std::ofstream out(final_path, std::ios::binary);
    if(!out.is_open()) {
        std::cerr << "Error: Cannot open file for writing: " << final_path << std::endl;
        return;
    }

    // 对于KITTI格式，直接写入点云数据，没有头信息
    for(const auto& point : *cloud) {
        float data[4] = {point.x, point.y, point.z, 1.0f}; // KITTI通常使用4值（x,y,z,反射率）
        out.write(reinterpret_cast<const char*>(data), sizeof(float) * 4);
    }
    
    out.close();
    std::cout << "Saved " << cloud->size() << " points to " << final_path << std::endl;
}