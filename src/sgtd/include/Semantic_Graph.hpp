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
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <Eigen/Geometry>
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
#include <boost/filesystem.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <chrono>
#include <ctime>
#include "cluster_manager.hpp"
#include "ndt_3d.h"
#include "matplotlibcpp.h"
#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)
#include <nlohmann/json.hpp>
#include "lapjav.hpp"
#include "linear_sum_assignment.hpp"
#include "ssc.h"
// #include "ssc.cpp"
#ifdef USE_VGICP_CUDA
#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif
#include <fast_gicp/gicp/fast_gicp.hpp>

namespace plt = matplotlibcpp; 
typedef std::vector<Eigen::Vector3f> Points;
typedef std::vector<int> IntVector;

// Structure to represent a Graph
struct Graph {
    std::vector<int> nodes;                    // List of nodes in the graph
    std::vector<Eigen::Vector2d> edges;        // List of edges (pairs of nodes)
    std::vector<float> weights;                // List of edge weights
    std::vector<float> volumes;                // List of volumes associated with nodes
    std::vector<float> densitys;               // List of densities associated with nodes
    std::vector<Eigen::Vector3f> centers;      // List of center points (3D coordinates)
    std::vector<float> poses;                  // List of poses (position or orientation)
    std::string path;                          // Path for saving or referencing the graph
    
    // Constructor to initialize the Graph structure
    Graph(const std::vector<int>& nodes,
          const std::vector<Eigen::Vector3f>& centers,
          const std::vector<float>& poses)
        : nodes(nodes), centers(centers), poses(poses) {}

    // Method to convert the Graph to a JSON format
    nlohmann::json toJSON() const {
        nlohmann::json j;
        
        // Add nodes to JSON
        j["nodes"] = nodes;
        
        // Add edges to JSON (converting Eigen vectors to standard double values)
        j["edges"] = std::vector<std::vector<double>>();
        for (const auto& edge : edges) {
            j["edges"].push_back({edge.x(), edge.y()});
        }
        
        // Add weights to JSON
        j["weights"] = weights;
        
        // Add centers to JSON (converting Eigen 3D vectors to float values)
        j["centers"] = std::vector<std::vector<float>>();
        for (const auto& center : centers) {
            j["centers"].push_back({center.x(), center.y(), center.z()});
        }
        
        // Add poses to JSON
        j["poses"] = poses;
        
        // Add volumes to JSON
        j["volumes"] = volumes;
        
        // Add densities to JSON
        j["densitys"] = densitys;
        
        return j;
    }
};

/**
 * @brief      Convert a JSON object to a Graph structure.
 * @param[in]  j: JSON object containing graph data
 * 
 * @return     A Graph object constructed from the JSON data
 * 
 * This function extracts various graph components (nodes, edges, weights, centers, poses, volumes, and densities) 
 * from the input JSON object and constructs a Graph object.
 */
Graph fromJSON(const nlohmann::json& j) {
    // Lambda function to convert a JSON array to std::vector<float>
    auto jsonToVector = [](const nlohmann::json& j) -> std::vector<float> {
        return j.get<std::vector<float>>();
    };

    // Lambda function to convert a JSON array to std::vector<Eigen::Vector2d>
    auto jsonToVector2d = [](const nlohmann::json& j) -> std::vector<Eigen::Vector2d> {
        std::vector<Eigen::Vector2d> vec;
        for (const auto& item : j) {
            vec.emplace_back(item[0].get<double>(), item[1].get<double>());
        }
        return vec;
    };

    // Lambda function to convert a JSON array to std::vector<Eigen::Vector3f>
    auto jsonToVector3f = [](const nlohmann::json& j) -> std::vector<Eigen::Vector3f> {
        std::vector<Eigen::Vector3f> vec;
        for (const auto& item : j) {
            vec.emplace_back(item[0].get<float>(), item[1].get<float>(), item[2].get<float>());
        }
        return vec;
    };

    // Extract data from JSON object
    std::vector<int> nodes = j["nodes"].get<std::vector<int>>();
    // std::vector<Eigen::Vector2d> edges = jsonToVector2d(j["edges"]);
    // std::vector<float> weights = jsonToVector(j["weights"]);
    std::vector<Eigen::Vector3f> centers = jsonToVector3f(j["centers"]);
    std::vector<float> poses = jsonToVector(j["poses"]);
    // std::vector<float> volumes = jsonToVector(j["volumes"]);
    // std::vector<float> densitys = jsonToVector(j["densitys"]);

    // Return the constructed Graph object
    return Graph(nodes, centers, poses);
};

/**
 * @brief      Read a Graph from a JSON file and convert it to a Graph structure.
 * @param[in]  filePath: The path to the JSON file containing graph data.
 * 
 * @return     A Graph object constructed from the JSON file content.
 * 
 * This function opens a JSON file from the specified file path, reads its content,
 * and uses the fromJSON function to convert the JSON data into a Graph object.
 * If the file cannot be opened, an exception will be thrown.
 */
Graph readGraphFromFile(const std::string& filePath) {
    // Open the JSON file
    std::ifstream inputFile(filePath);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Error opening file: " + filePath);
    }

    nlohmann::json j;
    inputFile >> j;  // Read the content of the JSON file

    // Close the file after reading
    inputFile.close();

    // Convert the JSON object to a Graph object using fromJSON
    return fromJSON(j);
};

/**
 * @brief      Get the current time using high resolution clock.
 * 
 * @return     Current time point as a `std::chrono::time_point`
 * 
 * This function returns the current time as a `std::chrono::time_point` using the 
 * high-resolution clock, which provides the most precise time measurement available.
 */
auto get_now_time(void) {
    return std::chrono::high_resolution_clock::now();
};

/**
 * @brief      Calculate the time difference between two time points.
 * @param[in]  a: The starting time point.
 * @param[in]  b: The ending time point.
 * 
 * @return     The difference between the two time points in milliseconds.
 * 
 * This function computes the time difference between two `std::chrono::time_point` objects 
 * (provided as parameters `a` and `b`). It returns the difference in milliseconds.
 */
template<typename T1, typename T2>
auto get_diff_time(const T1& a, const T2& b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
}



struct Semantic_Map {
    std::string bin_name ;
    Eigen::VectorXi dis_b ;
    double Sfeature;
    Eigen::Vector3f poses;
    Eigen::Matrix3f Rot;
    std::vector<Eigen::Vector3f> node_points;
    Eigen::MatrixXi dis_matrix;
    std::vector<float> densitys;
    std::vector<int> nodes;

    Semantic_Map(const std::string& bin_name,
                 const Eigen::VectorXi& dis_b,
                 const double& Sfeature,
                 const Eigen::Vector3f& poses,
                 const std::vector<Eigen::Vector3f>& node_points,
                 const Eigen::MatrixXi& dis_matrix,
                 const std::vector<int>& nodes,
                 const std::vector<float>& densitys)
        : bin_name(bin_name), dis_b(dis_b), Sfeature(Sfeature), poses(poses),node_points(node_points),dis_matrix(dis_matrix),nodes(nodes),densitys(densitys) {}

};

/**
 * @brief      Comparison functor for `Semantic_Map` objects based on their `Sfeature` value.
 * 
 * This struct defines a comparison operator `()` to be used with standard algorithms
 * or data structures like `std::sort` or `std::set`. It compares two `Semantic_Map`
 * objects by their `Sfeature` member.
 */
struct Compare_Semantic_Map {
    /**
     * @brief      Compare two `Semantic_Map` objects.
     * @param[in]  a: The first `Semantic_Map` object to compare.
     * @param[in]  b: The second `Semantic_Map` object to compare.
     * 
     * @return     `true` if `a.Sfeature` is less than `b.Sfeature`, otherwise `false`.
     * 
     * This operator is used to compare two `Semantic_Map` objects based on the value of their
     * `Sfeature` member. It returns `true` if the `Sfeature` of `a` is less than that of `b`,
     * and `false` otherwise. This can be useful for sorting or inserting objects into ordered
     * containers like `std::set`.
     */
    bool operator()(const Semantic_Map& a, const Semantic_Map& b) {
        return a.Sfeature < b.Sfeature;
    }
};


struct Match_pair{
    std::vector<int> matches;
    std::vector<int> node_i;
    std::vector<int> node_j;
    std::vector<Eigen::Vector3f> point_i;
    std::vector<Eigen::Vector3f> point_j;
    Eigen::Vector3f poses_i;
    Eigen::Vector3f poses_j;
    Eigen::Matrix3f Rot_i;
    Eigen::Matrix3f Rot_j;
    double Rmse = 1e8;
    double Rmse2=1e8;
    int fitness_iter = 0;
    Eigen::Matrix3f R = Eigen::Matrix3f::Zero();
    Eigen::Vector3f t = Eigen::Vector3f::Zero();
    double final_score ;
    std::string path_i;
    std::string path_j;
    Match_pair(const std::vector<int>& matches,
               const std::vector<Eigen::Vector3f>& point_i,
               const std::vector<Eigen::Vector3f>& point_j,
               const Eigen::Vector3f& poses_i,
               const Eigen::Vector3f& poses_j,
               const double& final_score )
        : matches(matches), point_i(point_i), point_j(point_j),poses_i(poses_i),poses_j(poses_j), final_score(final_score){}
    Match_pair(){};
};

/**
 * @brief      Compute the dot product of two 3D vectors.
 * @param[in]  a: The first 3D vector.
 * @param[in]  b: The second 3D vector.
 * 
 * @return     The dot product of the two vectors.
 * 
 * This function computes the dot product between two 3D vectors `a` and `b`.
 * The formula for the dot product is: 
 * \[ a \cdot b = a_0 * b_0 + a_1 * b_1 + a_2 * b_2 \]
 */
double vector_dot(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/**
 * @brief      Compute the magnitude (length) of a 3D vector.
 * @param[in]  a: The 3D vector.
 * 
 * @return     The magnitude of the vector.
 * 
 * This function computes the magnitude (length) of a 3D vector `a` using
 * the formula:
 * \[ \|a\| = \sqrt{a_0^2 + a_1^2 + a_2^2} \]
 */
double vector_magnitude(const Eigen::Vector3f& a) {
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

/**
 * @brief      Compute the angle (in degrees) between two 3D vectors.
 * @param[in]  a: The first 3D vector.
 * @param[in]  b: The second 3D vector.
 * 
 * @return     The angle (in degrees) between the two vectors.
 * 
 * This function calculates the angle between two 3D vectors `a` and `b`
 * using the dot product and magnitudes of the vectors. The formula for the
 * cosine of the angle \( \theta \) between the vectors is:
 * \[
 * \cos(\theta) = \frac{a \cdot b}{\|a\| \|b\|}
 * \]
 * The result is then converted from radians to degrees.
 */
double angle_between_vectors(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
    double dot = vector_dot(a, b);               // Compute the dot product
    double magA = vector_magnitude(a);           // Compute the magnitude of vector a
    double magB = vector_magnitude(b);           // Compute the magnitude of vector b
    double cosTheta = dot / (magA * magB);      // Compute the cosine of the angle
    return acos(cosTheta) * (180.0 / M_PI);     // Convert from radians to degrees and return
}


/**
 * @brief      Matches the current data graph with the semantic map data.
 * @param[in]  now_data: The current data graph containing nodes and associated data.
 * @param[in]  map_data: The semantic map data containing nodes and associated data.
 * @param[in]  BASE2OUSTER: Transformation matrix for converting base frame to ouster frame.
 * 
 * @return     A match pair object that contains matched nodes.
 * 
 * This function performs graph matching by comparing nodes between the
 * current data (`now_data`) and the semantic map (`map_data`). It uses
 * geometric features such as node positions and density, and computes 
 * matching costs to identify corresponding nodes. The result is returned 
 * as a match pair object.
 */
Match_pair graph_match(Graph now_data ,Semantic_Map map_data,Eigen::Matrix4f BASE2OUSTER)
{
    std::vector<int> node_i;
    std::vector<Eigen::Vector3f> point_i;
    std::vector<float> density_i;
    for (int i = 0; i < now_data.nodes.size();i++)
    {
        if(now_data.centers[i].norm() > 50)continue;

        node_i.push_back(now_data.nodes[i]);
        point_i.push_back(now_data.centers[i]);
        density_i.push_back(now_data.densitys[i]);
    }

    std::vector<int> node_j;
    std::vector<Eigen::Vector3f> point_j;
    std::vector<float> density_j;
    for (int i = 0; i < map_data.nodes.size();i++)
    {
        if(map_data.node_points[i].norm() > 50)continue;

        node_j.push_back(map_data.nodes[i]);
        point_j.push_back(map_data.node_points[i]);
        density_j.push_back(map_data.densitys[i]);
    }

    std::vector<Eigen::MatrixXd> point_i_feature(node_i.size());
    for(int i = 0; i < node_i.size();i++)
    {
        Eigen::MatrixXd node_i_matrix = Eigen::MatrixXd::Zero(9, 12);
        for(int j = 0; j < node_i.size();j++)
        {
            auto dis_ij = std::min(static_cast<int>(std::floor((point_i[i] - point_i[j]).norm()/5)), 11);
            node_i_matrix(node_i[j]-3,dis_ij)++;
        }
        point_i_feature.push_back(node_i_matrix);
    }

    std::vector<Eigen::MatrixXd> point_j_feature(node_j.size());
    for(int i = 0; i < node_j.size();i++)
    {
        Eigen::MatrixXd node_j_matrix = Eigen::MatrixXd::Zero(9, 12);
        for(int j = 0; j < node_j.size();j++)
        {
            auto dis_ij = std::min(static_cast<int>(std::floor((point_j[i] - point_j[j]).norm()/5)), 11);
            node_j_matrix(node_j[j]-3,dis_ij)++;
        }
        point_j_feature.push_back(node_j_matrix);
    }

    int matrix_size = std::max(node_i.size(), node_j.size());
    std::vector<std::vector<cost_t>> Match_matrix(node_i.size(), std::vector<cost_t>(node_j.size()));
    for (int i = 0; i < node_i.size();i++)
        for(int j = 0; j < node_j.size();j++)
            Match_matrix[i][j] = 1e8;
    for (int i = 0; i < node_i.size();i++)
    {
        
        for(int j = 0; j < node_j.size();j++)
        {
            if (node_i[i] != node_j[j])continue; // if (!((node_i[i] == 8 || node_i[i] == 10 ) &&  (node_j[j] == 8 || node_i[j] == 10 ))) 
            if (abs(density_i[i] - density_j[j])/density_i[i] > 3)continue;
            Eigen::MatrixXd Diff_ij = point_i_feature[i] - point_j_feature[j];
            Match_matrix[i][j] = 0;
            for (int z = 0;z < Diff_ij.rows();z++)
                Match_matrix[i][j] += static_cast<cost_t>(Diff_ij.row(z).norm());
        }
    }

    std::vector<int> matches(node_i.size(),-1);
    int num_rows = node_i.size();
    int num_cols = node_j.size();
    double cost_matrix[node_i.size()*node_j.size()];
    for (int i = 0; i < node_i.size();i++)
        for(int j = 0; j < node_j.size();j++)
            cost_matrix[i * node_j.size() + j ] = Match_matrix[i][j];

    bool maximize = false; 
    int64_t* a = nullptr;
    int64_t* b = nullptr;
    int ret;
    ret = linear_sum_assignment(cost_matrix, num_rows, num_cols, maximize, &a, &b);

    int size_ret = num_rows < num_cols ? num_rows : num_cols ;
    if (ret == 0) {
        // std::cout << "Assignment result:" << std::endl;
        for (int i =0; i < size_ret;i++)
        {
            // if (row_ind[i] >=  node_j.size()) continue;
            if (Match_matrix[a[i]][b[i]] < 1e8)
            {
                matches[a[i]] = b[i];
            }
        }
        delete[] a;
        delete[] b;
    } else if (ret == RECTANGULAR_LSAP_INFEASIBLE) {
        std::cout << "Cost matrix is infeasible" << std::endl;
    } else if (ret == RECTANGULAR_LSAP_INVALID) {
        std::cout << "Matrix contains invalid numeric entries" << std::endl;
    } else {
        std::cout << "An error occurred" << std::endl;
    }

    for(int i = 0 ;i < matches.size();i++)
    {
        if (matches[i] == -1)continue;

        std::vector<Eigen::Vector3f> trangle_list_i,trangle_list_match;
        double score_num = 0.0;

        for(int j = 0 ;j < matches.size();j++)
        {
            if (matches[j] == -1 || i==j )continue;

            Eigen::Vector3f diff_i((point_i[i]-point_i[j]).norm(),0,0);
            Eigen::Vector3f diff_match((point_j[matches[i]]-point_j[matches[j]]).norm(), \
                                        0, \
                                        0);
            trangle_list_i.push_back(diff_i);
            trangle_list_match.push_back(diff_match);
        }
        if (trangle_list_i.size() > 0 && trangle_list_match.size() > 0)
        {
            for (int j = 0 ;j < trangle_list_i.size();j++)
            {
                if ((trangle_list_i[j] - trangle_list_match[j]).norm() < (trangle_list_i[j]).norm() * 0.1) score_num++; //(trangle_list_i[j]).norm() * 0.1
            }
            if (score_num/((double)(trangle_list_i.size())) < 0.2) matches[i] = -1;
        }
    }


    double final_score = 0;
    for (int i =0; i < node_i.size();i++)if (matches[i] == -1)final_score ++;
    final_score = 1 - (final_score/node_i.size());

    Eigen::Vector3f poses_i(now_data.poses[3],now_data.poses[7],now_data.poses[11]);
    Eigen::Vector3f poses_j=map_data.poses;
    Eigen::Matrix3f Rot ;
    Rot << now_data.poses[0] ,now_data.poses[1] ,now_data.poses[2],
           now_data.poses[4] ,now_data.poses[5] ,now_data.poses[6],
           now_data.poses[8] ,now_data.poses[9] ,now_data.poses[10];

    Eigen::Matrix4f transform_i = Eigen::Matrix4f::Identity();
    transform_i.block<3, 3>(0, 0) = Rot;
    transform_i.block<3, 1>(0, 3) = poses_i;

    transform_i = transform_i * BASE2OUSTER;
    Rot = transform_i.block<3, 3>(0, 0);
    poses_i = transform_i.block<3, 1>(0, 3);

    Match_pair this_Match_pair(matches,point_i,point_j,poses_i,poses_j,final_score);
    this_Match_pair.Rot_i = Rot;
    this_Match_pair.Rot_j = map_data.Rot;
    this_Match_pair.path_i  = now_data.path;
    this_Match_pair.path_j  = map_data.bin_name;
    this_Match_pair.node_i = node_i;
    this_Match_pair.node_j = node_j;

    return this_Match_pair;

}

/**
 * @brief      Compares two `Match_pair` objects based on their final score.
 * @param[in]  a: The first `Match_pair` object.
 * @param[in]  b: The second `Match_pair` object.
 * 
 * @return     `true` if the final score of `a` is greater than that of `b`, otherwise `false`.
 * 
 * This function is used to compare two match pairs by their final score.
 * It is typically used for sorting match pairs in descending order of their final score.
 */
bool Compare_Semantic_Match_pair_final_score(Match_pair a, Match_pair b) {
    return a.final_score > b.final_score;  // Return true if 'a' has a greater final score than 'b'
}

/**
 * @brief      Compares two `Match_pair` objects based on their RMSE (Root Mean Square Error).
 * @param[in]  a: The first `Match_pair` object.
 * @param[in]  b: The second `Match_pair` object.
 * 
 * @return     `true` if the RMSE of `a` is smaller than that of `b`, otherwise `false`.
 * 
 * This function is used to compare match pairs by their RMSE value.
 * It is commonly used for sorting match pairs in ascending order of their RMSE.
 */
bool Compare_Semantic_Match_pair_Rmse(Match_pair a, Match_pair b) {
    return a.Rmse < b.Rmse;  // Return true if 'a' has a smaller RMSE than 'b'
}



bool Compare_Semantic_map_sort(Semantic_Map a,Semantic_Map b){return a.Sfeature < b.Sfeature;};


class RvizViewFollower {
private:
    ros::NodeHandle nh_;
    tf::TransformBroadcaster tf_broadcaster_;
    ros::Publisher camera_marker_pub_;
    
    double camera_distance_;
    double camera_height_offset_;
    bool last_pose_valid_;
    Eigen::Matrix4f last_pose_matrix_;
    std::string fixed_frame_;
    bool follow_orientation_;  // 是否跟随目标物体的角度

public:
    RvizViewFollower(ros::NodeHandle& nh) : 
        nh_(nh),
        camera_distance_(15.0),
        camera_height_offset_(5.0),
        last_pose_valid_(false),
        fixed_frame_("map"),
        follow_orientation_(true)  // 默认启用角度跟随
    {
        // 获取参数
        nh_.param<double>("rviz_view_follower/camera_distance", camera_distance_, 15.0);
        nh_.param<double>("rviz_view_follower/camera_height_offset", camera_height_offset_, 5.0);
        nh_.param<std::string>("rviz_view_follower/fixed_frame", fixed_frame_, "map");
        nh_.param<bool>("rviz_view_follower/follow_orientation", follow_orientation_, true);
        
        // 初始化相机标记发布者
        camera_marker_pub_ = nh_.advertise<visualization_msgs::Marker>(
            "/rviz_camera_position", 10);
        
        ROS_INFO("RViz View Follower initialized. Orientation follow: %s", 
                 follow_orientation_ ? "enabled" : "disabled");
    }
    
    // 使用4×4矩阵设置相机视角
    void setCameraFromMatrix(const Eigen::Matrix4f& pose_matrix) {
        // 检查矩阵是否有效
        if (!isMatrixValid(pose_matrix)) {
            ROS_WARN("Invalid pose matrix provided to camera follower");
            return;
        }
        
        // 记录最后一个有效位姿
        last_pose_matrix_ = pose_matrix;
        last_pose_valid_ = true;
        
        // 提取旋转矩阵和位置
        Eigen::Matrix3f rotation = pose_matrix.block<3, 3>(0, 0);
        Eigen::Vector3f position(pose_matrix(0, 3), pose_matrix(1, 3), pose_matrix(2, 3));
        
        // 计算相机位置和朝向
        Eigen::Vector3f camera_position;
        Eigen::Matrix3f camera_rotation;
        
        if (follow_orientation_) {
            // 使用目标物体的方向，为相机创建新的朝向
            // 假设物体的X轴是前向，相机需要从后方看向物体
            Eigen::Vector3f target_forward = rotation.col(0);  // 物体的前向方向（X轴）
            Eigen::Vector3f target_up = rotation.col(2);       // 物体的上方向（Z轴）
            
            // 相机需要从目标物体的反方向看向物体
            Eigen::Vector3f camera_forward = -target_forward;
            Eigen::Vector3f camera_up = target_up;
            
            // 计算相机的右向量，确保正交性
            Eigen::Vector3f camera_right = camera_up.cross(camera_forward).normalized();
            
            // 重新计算up向量，确保三个向量互相垂直
            camera_up = camera_forward.cross(camera_right).normalized();
            
            // 构建相机的旋转矩阵
            camera_rotation.col(0) = camera_forward;
            camera_rotation.col(1) = camera_right;
            camera_rotation.col(2) = camera_up;
            
            // 计算相机位置 - 在目标后方一定距离，并有一定高度偏移
            camera_position = position - target_forward * camera_distance_ + 
                              target_up * camera_height_offset_;
        } else {
            // 简单的位置跟随，不跟随方向
            // 相机位置始终在目标的后方（-X方向）
            camera_position = position + Eigen::Vector3f(-camera_distance_, 0, camera_height_offset_);
            
            // 相机旋转设置为基本的朝向（面向+X方向）
            camera_rotation = Eigen::Matrix3f::Identity();
        }
        
        // 广播TF用于RViz跟随
        broadcastCameraTF(camera_position, camera_rotation);
        
        // 发布可视化标记
        publishCameraMarker(camera_position, camera_rotation);
    }
    
    // 发送TF变换
    void broadcastCameraTF(const Eigen::Vector3f& position, const Eigen::Matrix3f& rotation) {
        tf::Transform camera_tf;
        
        // 设置位置
        camera_tf.setOrigin(tf::Vector3(position(0), position(1), position(2)));
        
        // 从旋转矩阵转换为四元数
        Eigen::Quaternionf q(rotation);
        camera_tf.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
        
        // 广播TF
        tf_broadcaster_.sendTransform(tf::StampedTransform(
            camera_tf, ros::Time::now(), fixed_frame_, "rviz_camera"));
    }
    
    // 发布相机标记
    void publishCameraMarker(const Eigen::Vector3f& position, const Eigen::Matrix3f& rotation) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = fixed_frame_;
        marker.header.stamp = ros::Time::now();
        marker.ns = "rviz_camera";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::ARROW;
        marker.action = visualization_msgs::Marker::ADD;
        
        // 设置标记位置
        marker.pose.position.x = position(0);
        marker.pose.position.y = position(1);
        marker.pose.position.z = position(2);
        
        // 从旋转矩阵转换为四元数
        Eigen::Quaternionf q(rotation);
        marker.pose.orientation.x = q.x();
        marker.pose.orientation.y = q.y();
        marker.pose.orientation.z = q.z();
        marker.pose.orientation.w = q.w();
        
        // 设置标记大小
        marker.scale.x = 3.0;  // 箭头长度
        marker.scale.y = 0.5;  // 箭头宽度
        marker.scale.z = 0.5;  // 箭头高度
        
        // 设置标记颜色
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;
        
        marker.lifetime = ros::Duration(0.1);  // 短寿命，确保持续更新
        
        camera_marker_pub_.publish(marker);
    }
    
    // 检查矩阵是否有效
    bool isMatrixValid(const Eigen::Matrix4f& matrix) {
        // 检查是否包含NaN或无穷大值
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (std::isnan(matrix(i, j)) || std::isinf(matrix(i, j))) {
                    return false;
                }
            }
        }
        
        // 确保旋转部分是有效的旋转矩阵
        Eigen::Matrix3f rotation = matrix.block<3, 3>(0, 0);
        float det = rotation.determinant();
        if (std::abs(det - 1.0f) > 0.1f) {
            return false;
        }
        
        return true;
    }
    
    // 设置是否跟随目标的方向
    void setFollowOrientation(bool enable) {
        follow_orientation_ = enable;
        ROS_INFO("Orientation follow %s", enable ? "enabled" : "disabled");
        
        // 如果有有效的上一个位姿，更新相机视角
        if (last_pose_valid_) {
            setCameraFromMatrix(last_pose_matrix_);
        }
    }
    
    // 设置相机距离
    void setCameraDistance(double distance) {
        camera_distance_ = distance;
        // 如果有有效的上一个位姿，更新相机视角
        if (last_pose_valid_) {
            setCameraFromMatrix(last_pose_matrix_);
        }
    }
    
    // 设置相机高度偏移
    void setCameraHeightOffset(double offset) {
        camera_height_offset_ = offset;
        // 如果有有效的上一个位姿，更新相机视角
        if (last_pose_valid_) {
            setCameraFromMatrix(last_pose_matrix_);
        }
    }
};

// 全局对象，便于在您的程序各处访问
RvizViewFollower* g_view_follower = nullptr;

// 获取全局视角跟随器
RvizViewFollower* getViewFollower() {
    return g_view_follower;
}

// 初始化函数
void initViewFollower(ros::NodeHandle& nh) {
    if (g_view_follower == nullptr) {
        g_view_follower = new RvizViewFollower(nh);
    }
}

// 清理函数
void cleanupViewFollower() {
    if (g_view_follower != nullptr) {
        delete g_view_follower;
        g_view_follower = nullptr;
    }
}