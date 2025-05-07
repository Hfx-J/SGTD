#ifndef STDESC_H
#define STDESC_H

#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <pcl/common/io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <stdio.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

#include "omp.h"

// adding for slict structure
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#define HASH_P 116101
#define MAX_N 10000000000
#define MAX_FRAME_N 20000




typedef struct ConfigSetting {
    /* for point cloud pre-preocess*/
    int stop_skip_enable_ = 0;
    double ds_size_ = 0.5;
    int maximum_corner_num_ = 30;

    /* for key points*/
    double plane_merge_normal_thre_;
    double plane_merge_dis_thre_;
    double plane_detection_thre_ = 0.01;
    double voxel_size_ = 1.0;
    int voxel_init_num_ = 10;
    double proj_image_resolution_ = 0.5;
    double proj_dis_min_ = 0.2;
    double proj_dis_max_ = 5;
    double corner_thre_ = 10;

    /* for STD */
    int descriptor_near_num_ = 10;
    double descriptor_min_len_ = 1;
    double descriptor_max_len_ = 10;
    double non_max_suppression_radius_ = 3.0;
    double std_side_resolution_ = 0.2;

    /* for place recognition*/
    int skip_near_num_ = 50;
    int candidate_num_ = 50;
    int sub_frame_num_ = 10;
    double rough_dis_threshold_ = 0.03;
    double vertex_diff_threshold_ = 0.7;
    double icp_threshold_ = 0.5;
    double normal_threshold_ = 0.1;
    double dis_threshold_ = 0.3;

} ConfigSetting;

// Structure for Stabel Triangle Descriptor
typedef struct STDesc {
    // the side lengths of STDesc, arranged from short to long
    Eigen::Vector3d side_length_;

    // projection angle between vertices
    Eigen::Vector3d angle_;

    Eigen::Vector3d center_;
    unsigned int frame_id_;

    // three vertexs
    Eigen::Vector3d vertex_A_;
    Eigen::Vector3d vertex_B_;
    Eigen::Vector3d vertex_C_;

    // some other inform attached to each vertex,e.g., intensity
    Eigen::Vector3d vertex_attached_;
    // covariance matrix computed by neighboring points
    std::vector<int> node_id; 
    Eigen::Matrix3d cov_mat_A_;
    Eigen::Matrix3d cov_mat_B_;
    Eigen::Matrix3d cov_mat_C_;
} STDesc;

struct LOOP_RESULT{
    int match_id;
    int match_fitness;
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
    std::vector<std::pair<STDesc, STDesc>> loop_std_pair;
};
// plane structure for corner point extraction
typedef struct Plane {
    pcl::PointXYZINormal p_center_;
    Eigen::Vector3d center_;
    Eigen::Vector3d normal_;
    Eigen::Matrix3d covariance_;
    float radius_ = 0;
    float min_eigen_value_ = 1;
    float intercept_ = 0;
    int id_ = 0;
    int sub_plane_num_ = 0;
    int points_size_ = 0;
    bool is_plane_ = false;
} Plane;

typedef struct STDMatchList {
    std::vector<std::pair<STDesc, STDesc>> match_list_;
    std::pair<int, int> match_id_;
    double mean_dis_;
} STDMatchList;

class VOXEL_LOC {
public:
    int64_t x, y, z;

    VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
        : x(vx), y(vy), z(vz) {}

    bool operator==(const VOXEL_LOC &other) const {
        return (x == other.x && y == other.y && z == other.z);
    }
};

// for down sample function
struct M_POINT {
    float xyz[3];
    float intensity;
    int count = 0;
};

// Hash value

template <>
struct std::hash<VOXEL_LOC> {
    int64_t operator()(const VOXEL_LOC &s) const {
        using std::hash;
        using std::size_t;
        return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
    }
};

/////////////////// begin slict defined types////////////////////

struct PointTQXYZI {
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;  // preferred way of adding a XYZ+padding
    double t;
    float qx;
    float qy;
    float qz;
    float qw;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are aligned
} EIGEN_ALIGN16;  // enforce SSE padding for correct memory alignment
POINT_CLOUD_REGISTER_POINT_STRUCT(
        PointTQXYZI,
        (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(
                double, t, t)(float, qx, qx)(float, qy, qy)(float,
                                                            qz,
                                                            qz)(float, qw, qw))
typedef PointTQXYZI PointPose;
typedef pcl::PointCloud<PointPose> CloudPose;
typedef pcl::PointCloud<PointPose>::Ptr CloudPosePtr;

/////////////////// end slict defined types////////////////////
class Point_Pair {
public:
    int64_t x, y, z, a, b, c;

    Point_Pair(int64_t vx = 0,
               int64_t vy = 0,
               int64_t vz = 0,
               int64_t va = 0,
               int64_t vb = 0,
               int64_t vc = 0)
        : x(vx), y(vy), z(vz), a(va), b(vb), c(vc) {}

    bool operator==(const Point_Pair &other) const {
        return (x == other.x && y == other.y && z == other.z && a == other.a &&
                b == other.b && c == other.c);
    }
};

template <>
struct std::hash<Point_Pair> {
    int64_t operator()(const Point_Pair &s) const {
        using std::hash;
        using std::size_t;
        // return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
        return ((((((((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N +
                     (s.x)) *
                    HASH_P) %
                           MAX_N +
                   s.a) *
                  HASH_P) %
                         MAX_N +
                 s.b) *
                HASH_P) %
                       MAX_N +
               s.c;
    }
};

class STDesc_LOC {
public:
    int64_t x, y, z, a, b, c;

    STDesc_LOC(int64_t vx = 0,
               int64_t vy = 0,
               int64_t vz = 0,
               int64_t va = 0,
               int64_t vb = 0,
               int64_t vc = 0)
        : x(vx), y(vy), z(vz), a(va), b(vb), c(vc) {}

    bool operator==(const STDesc_LOC &other) const {
        // use three attributes
        // return (x == other.x && y == other.y && z == other.z);
        // use six attributes
        // return (x == other.x && y == other.y && z == other.z && a == other.a
        // && b == other.b && c == other.c);
        return (x == other.x && y == other.y && z == other.z && a == other.a);
    }
};

template <>
struct std::hash<STDesc_LOC> {
    int64_t operator()(const STDesc_LOC &s) const {
        using std::hash;
        using std::size_t;
        // return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
        return ((((((s.z * HASH_P) % MAX_N + s.y) * HASH_P) % MAX_N + s.x) * HASH_P) % MAX_N + s.a);

        // return ((((((((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N +
        //  (s.x)) * HASH_P) % MAX_N + s.a) * HASH_P) % MAX_N + s.b) * HASH_P) %  MAX_N + s.c;
    }
};

class SG_STDesc_LOC {
public:
    int64_t x, y, z, a, b, c;

    SG_STDesc_LOC(int64_t vx = 0,
                int64_t vy = 0,
                int64_t vz = 0,
                int64_t va = 0,
                int64_t vb = 0,
                int64_t vc = 0)
        : x(vx), y(vy), z(vz), a(va), b(vb), c(vc) {}

    bool operator==(const SG_STDesc_LOC &other) const {
        // use three attributes
        return (x == other.x && y == other.y && z == other.z);
        // use six attributes
        // return (x == other.x && y == other.y && z == other.z && a == other.a
        // && b == other.b && c == other.c);
    }
};

template <>
struct std::hash<SG_STDesc_LOC> {
    int64_t operator()(const SG_STDesc_LOC &s) const {
        using std::hash;
        using std::size_t;
        return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
        // return ((((((((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N +
        //  (s.x)) * HASH_P) % MAX_N + s.a) * HASH_P) % MAX_N + s.b) * HASH_P) %  MAX_N + s.c;
    }
};
// OctoTree structure for plane detection
class OctoTree {
public:
    ConfigSetting config_setting_;
    std::vector<Eigen::Vector3d> voxel_points_;
    Plane *plane_ptr_;
    int layer_;
    int octo_state_;  // 0 is end of tree, 1 is not
    int merge_num_ = 0;
    bool is_project_ = false;
    std::vector<Eigen::Vector3d> proj_normal_vec_;

    // check 6 direction: x,y,z,-x,-y,-z
    bool is_check_connect_[6];
    bool connect_[6];
    OctoTree *connect_tree_[6];

    bool is_publish_ = false;
    OctoTree *leaves_[8];
    double voxel_center_[3];  // x, y, z
    float quater_length_;
    bool init_octo_;
    OctoTree(const ConfigSetting &config_setting)
        : config_setting_(config_setting) {
        voxel_points_.clear();
        octo_state_ = 0;
        layer_ = 0;
        init_octo_ = false;
        for (int i = 0; i < 8; i++) {
            leaves_[i] = nullptr;
        }
        for (int i = 0; i < 6; i++) {
            is_check_connect_[i] = false;
            connect_[i] = false;
            connect_tree_[i] = nullptr;
        }
        plane_ptr_ = new Plane;
    }
    void init_plane();
    void init_octo_tree();
};

void down_sampling_voxel(pcl::PointCloud<pcl::PointXYZI> &pl_feat,
                         double voxel_size);

void load_pose_with_time(
        const std::string &pose_file,
        std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &poses_vec,
        std::vector<double> &times_vec);



void read_parameters(ros::NodeHandle &nh, ConfigSetting &config_setting);

double time_inc(std::chrono::_V2::system_clock::time_point &t_end,
                std::chrono::_V2::system_clock::time_point &t_begin);

bool attach_greater_sort(std::pair<double, int> a, std::pair<double, int> b);

class STDescManager {
public:
    STDescManager() = default;

    ConfigSetting config_setting_;

    int CS1;

    unsigned int current_frame_id_;

    Eigen::Matrix3Xd src_points_;
    Eigen::Matrix3Xd tgt_points_;

    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
            src_covariances_matched_;
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
            tgt_covariances_matched_;

    std::vector<STDesc> matched_stds_;

    STDescManager(ConfigSetting &config_setting)
        : config_setting_(config_setting) {
        current_frame_id_ = 0;

        current_plane_cloud_.reset(new pcl::PointCloud<pcl::PointXYZINormal>);
    };

    // hash table, save all descriptors
    std::unordered_map<STDesc_LOC, std::vector<STDesc>> data_base_;

    // save all key clouds, optional
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> key_cloud_vec_;

    // save all corner points, optional
    std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> corner_cloud_vec_;

    // save all planes of key frame, required
    std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> plane_cloud_vec_;

    // save current plane cloud for relocalization purpose, required;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr current_plane_cloud_;

    /*Three main processing functions*/

    // generate STDescs from a point cloud
    void GenerateSTDescs(pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                         std::vector<STDesc> &stds_vec);


    /**
     * @brief build semantic std descriptor for the single scan
     *
     * @param instance_pc
     * @param stds_vec
     */
    void BuildSingleScanSTD(
            const pcl::PointCloud<pcl::PointXYZL>::Ptr &instance_pc,
            std::vector<STDesc> &stds_vec);


    // search result <candidate_id, plane icp score>. -1 for no loop
    void SearchLoop(const std::vector<STDesc> &stds_vec,
                    std::pair<int, double> &loop_result,
                    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &loop_transform,
                    std::vector<std::pair<STDesc, STDesc>> &loop_std_pair,
                    std::vector<LOOP_RESULT> &match_result_list);

    // add descriptors to database
    void AddSTDescs(const std::vector<STDesc> &stds_vec);



private:
    /*Following are sub-processing functions*/
    // build STDescs from corner points.
    void build_stdesc(
            const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points,
            std::vector<STDesc> &stds_vec);


    // Select a specified number of candidate frames according to the
    // number of STDesc rough matches
    void candidate_selector(const std::vector<STDesc> &stds_vec,
                            std::vector<STDMatchList> &candidate_matcher_vec);

    // Get the best candidate frame by geometry check
    void candidate_verify(
            const STDMatchList &candidate_matcher,
            double &verify_score,
            std::pair<Eigen::Vector3d, Eigen::Matrix3d> &relative_pose,
            std::vector<std::pair<STDesc, STDesc>> &sucess_match_vec);

    // Get the transform between a matched std pair
    void triangle_solver(std::pair<STDesc, STDesc> &std_pair,
                         Eigen::Vector3d &t,
                         Eigen::Matrix3d &rot);


};
void publish_std(const std::vector<std::pair<STDesc, STDesc>> &match_std_list,
                 const Eigen::Matrix4d &transform1,
                 const Eigen::Matrix4d &transform2,
                 const ros::Publisher &std_publisher);
#endif  // STDESC_H