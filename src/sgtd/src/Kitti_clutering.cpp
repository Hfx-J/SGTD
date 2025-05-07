
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
// #include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
// #include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
// #include "ndt_3d.h"
#include <yaml-cpp/yaml.h>
#include <unordered_map>
#include <boost/filesystem.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <chrono>
#include <ctime>
// #include "FEC.h"
#include "cluster_manager.hpp"
// #include "ndt_3d.h"
#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

#include <nlohmann/json.hpp>

struct Graph {
    std::vector<int> nodes;
    std::vector<Eigen::Vector2d> edges;
    std::vector<float> weights;
    std::vector<float> volumes;
    std::vector<float> densitys;
    std::vector<Eigen::Vector3f> centers;
    std::vector<float> poses;
    Graph(const std::vector<int>& nodes,
          const std::vector<Eigen::Vector2d>& edges,
          const std::vector<float>& weights,
          const std::vector<Eigen::Vector3f>& centers,
          const std::vector<float>& poses,
          const std::vector<float>& volumes,
          const std::vector<float>& densitys)
        : nodes(nodes), edges(edges), weights(weights), centers(centers),poses(poses),volumes(volumes),densitys(densitys) {}

    // 将Graph转换为JSON对象
    nlohmann::json toJSON() const {
        nlohmann::json j;
        j["nodes"] = nodes;
        j["edges"] = std::vector<std::vector<double>>();
        for (const auto& edge : edges) {
            j["edges"].push_back({edge.x(), edge.y()});
        }
        j["weights"] = weights;
        j["centers"] = std::vector<std::vector<float>>();
        for (const auto& center : centers) {
            j["centers"].push_back({center.x(), center.y(), center.z()});
        }
        j["poses"] = poses;
        j["volumes"] = volumes;
        j["densitys"] = densitys;
        return j;
    }
};

struct Graph_map {
    std::vector<int> nodes;
    std::vector<Eigen::Vector2d> edges;
    std::vector<float> weights;
    std::vector<Eigen::Vector3f> centers;
    std::vector<Eigen::Vector3f> poses;
    Graph_map(const std::vector<int>& nodes,
          const std::vector<Eigen::Vector2d>& edges,
          const std::vector<float>& weights,
          const std::vector<Eigen::Vector3f>& centers,
          const std::vector<Eigen::Vector3f>& poses)
        : nodes(nodes), edges(edges), weights(weights), centers(centers) , poses(poses){}

    // 将Graph转换为JSON对象
    nlohmann::json toJSON() const {
        nlohmann::json j;
        j["nodes"] = nodes;
        j["edges"] = std::vector<std::vector<double>>();
        for (const auto& edge : edges) {
            j["edges"].push_back({edge.x(), edge.y()});
        }
        j["weights"] = weights;
        j["centers"] = std::vector<std::vector<float>>();
        for (const auto& center : centers) {
            j["centers"].push_back({center.x(), center.y(), center.z()});
        }
        j["poses"] = std::vector<std::vector<float>>();
        for (const auto& pose : poses) {
            j["poses"].push_back({pose.x(), pose.y(), pose.z()});
        }
        return j;
    }
};

std::string save_velo_dir;
std::ofstream pose_file;
std::ofstream path_file;
std::ifstream path_file_read;
std::ofstream gnsspath_file;
std::ofstream timestamp_file;

std::unordered_map<int, int> learning_map = {
    {0, 0}, 
    {1, 0}, 
    {10, 0},
    {11, 2}, 
    {13, 0}, 
    {15, 3}, 
    {16, 0}, 
    {18, 0}, 
    {20, 0}, 
    {30, 6}, 
    {31, 7}, 
    {32, 8}, 
    {40, 9},
    {44, 10}, 
    {48, 11}, 
    {49, 12}, 
    {50, 13}, 
    {51, 14}, 
    {52, 0}, 
    {60, 9}, 
    {70, 15}, 
    {71, 16}, 
    {72, 17}, 
    {80, 18}, 
    {81, 19},
    {99, 0}, 
    {252, 0}, 
    {253, 7}, 
    {254, 6}, 
    {255, 8}, 
    {256, 0}, 
    {257, 0}, 
    {258, 0}, 
    {259, 0}
};

int max_key = 259; // 根据learning_map中的最大键值确定
std::vector<int> remap_lut(max_key + 100, 0); // 初始化LUT
// std::unordered_map<int, int> node_map = {
//     {1, 0}, {4, 1}, {5, 2}, {11, 3}, {12, 4}, {13, 5}, {14, 6}, {15, 7}, {16, 8}, {17, 9}, {18, 10}, {19, 11}
// };
std::unordered_map<int, int> node_map = { // spnvas
    {0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {10, 3}, {11, 4}, {12, 5}, {13, 6}, {14, 7}, {15, 8}, {16, 9}, {17, 10}, {18, 11}
};
// train_label_name_mapping = {
//         0: 'car',
//         1: 'bicycle',
//         2: 'motorcycle',
//         3: 'truck',
//         4: 'other-vehicle',
//         5: 'person',
//         6: 'bicyclist',
//         7: 'motorcyclist',
//         8: 'road',
//         9: 'parking',
//         10: 'sidewalk',
//         11: 'other-ground',
//         12: 'building',
//         13: 'fence',
//         14: 'vegetation',
//         15: 'trunk',
//         16: 'terrain',
//         17: 'pole',
//         18: 'traffic-sign'
//     }
std::mutex cluster_mutex;  // 用于保护`cluster`和`inst_id`
std::vector<Eigen::VectorXf> cluster;
int inst_id = 0;




/*MY Funcation*/

// void gen_map(std::vector<std::string> scan_names,std::vector<std::string> label_names,std::string poses_dir,std::string label_output_dir,std::string seq_path)
// {
//     std::vector<Eigen::Vector4f> cloud_map;
//     std::vector<Eigen::Vector3f> cloud_poses;
//     std::vector<int> cloud_label;
//     std::string line;
//     std::ifstream file(poses_dir); // 假设pose.txt是你的文件名
//     std::cout<<"merge start!"<<std::endl;
//     for (int map_i=0;map_i<scan_names.size();map_i++)
//     {
//         std::string scan_name = scan_names[map_i];
//         auto start = std::chrono::high_resolution_clock::now();

//         getline(file, line);
//         std::istringstream iss(line);
//         std::vector<float> poseRow;
//         float value;
//         while (iss >> value) {
//             poseRow.push_back(value);
//         }
//         Eigen::Vector3f ps(poseRow[3],poseRow[7],poseRow[11]);
//         cloud_poses.push_back(ps);
//         Eigen::Matrix3f pose_map = Eigen::Matrix3f::Identity();
//         Eigen::Vector3f T_map(poseRow[3],poseRow[7],poseRow[11]);
//         // for(int mat_i = 0;mat_i<4;mat_i++)
//         //     for(int mat_j=0;mat_j<4;mat_j++)
//         //         pose_map(mat_i,mat_j) = 0;
//         // pose_map(3,3)=1;    
//         for(int mat_i = 0;mat_i<3;mat_i++)
//             for(int mat_j=0;mat_j<3;mat_j++)
//                 pose_map(mat_i,mat_j) = poseRow[mat_i*4 +mat_j];
        
//         //std::cout<<"merge 1 start!"<<std::endl;
//     // auto gen_start = std::chrono::high_resolution_clock::now();
//         // 打开二进制点云文件
//         std::ifstream scan_file(scan_name, std::ios::binary);
//         // 读取数据到vector中
//         //std::vector<float> scan_data((std::istreambuf_iterator<char>(scan_file)), std::istreambuf_iterator<char>());
//         scan_file.seekg(0, std::ios::end);
//         size_t size = scan_file.tellg();
//         scan_file.seekg(0, std::ios::beg);

//         // 根据文件大小和浮点数大小计算点的数量
//         size_t num_points = size / sizeof(float);

//         // 读取浮点数到vector中
//         std::vector<float> scan_data(num_points);
//         scan_file.read(reinterpret_cast<char*>(scan_data.data()), size);

//         // 将数据映射到Eigen矩阵
//         Eigen::Map<Eigen::MatrixXf> points(scan_data.data(), 4, num_points / 4);

//         // Eigen::MatrixXf points_transposed = points.transpose();
//         // Eigen::MatrixXf transformed_points = (pose_map * points) + T_map;

//         // 更新原始点云数据
//         // points = Eigen::Map<Eigen::MatrixXf>(transformed_points.data(), points.rows(), points.cols());

//         // 使用Eigen的Map将数据映射为矩阵，方便处理。这里假设每个点有4个浮点数（x, y, z, intensity）
//         // 计算每个点的角度
//         //Eigen::VectorXf angles = points.row(1).binaryExpr(points.row(0), static_cast<float(*)(float, float)>(std::atan2));

//         std::ifstream label_file(label_names[map_i], std::ios::binary);
//         std::vector<uint32_t> label_data;

//         label_file.seekg(0, std::ios::end);
//         size_t size_la = label_file.tellg();
//         label_file.seekg(0, std::ios::beg);

//         // 根据文件大小调整vector大小（每个uint32_t占4字节）
//         label_data.resize(size_la / sizeof(uint32_t));

//         // 直接读取到vector的内存中
//         label_file.read(reinterpret_cast<char*>(label_data.data()), size_la);

//         // 将标签数据映射为Eigen整数向量
//         Eigen::Map<Eigen::VectorXi> labels(reinterpret_cast<int*>(label_data.data()), label_data.size());
//         // std::cout << " label " << labels<< std::endl;
//         // 之后可以进行其他处理...
//     // std::cout << "gen_labels 2" << std::endl;
//         assert(points.cols()==labels.size());
//         //std::cout<<"merge 2 start!"<<std::endl;

//         for(int i = 0;i<labels.size();i++)
//         {   
//             Eigen::Vector4f p4_inst = points.col(i);
//             Eigen::Vector3f p3_inst(p4_inst(0),p4_inst(1),p4_inst(2));
//             Eigen::Vector3f pp = pose_map*p3_inst + T_map;
//             Eigen::Vector4f ppp(pp(0),pp(1),pp(2),1);
//             cloud_map.push_back(ppp);
//             cloud_label.push_back(labels[i]);
//         }
//     }

//     std::vector<Eigen::Vector4f> cloud_map_inst;
//     std::vector<int> cloud_label_inst;
//     for(int inst_i=0;inst_i<cloud_label.size();inst_i+=10)
//     {
//         cloud_map_inst.push_back(cloud_map[inst_i]);
//         cloud_label_inst.push_back(cloud_label[inst_i]);
//     }
//     cloud_map.clear();
//     cloud_label.clear();
//     cloud_map = cloud_map_inst;
//     cloud_label = cloud_label_inst;
//     cloud_label_inst.clear();
//     cloud_map_inst.clear();

//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//     cloud->width = cloud_map.size(); // 设置点云数量
//     cloud->height = 1; // 非有序点云
//     cloud->is_dense = false; // 有无效点
//     cloud->points.resize(cloud->width * cloud->height);

//     for (size_t i = 0; i < cloud_map.size(); ++i) {
//         cloud->points[i].x = cloud_map[i][0] ;
//         cloud->points[i].y = cloud_map[i][1];
//         cloud->points[i].z = cloud_map[i][2];
//     }
//     if (pcl::io::savePCDFileASCII("/media/beabbit/T5 EVO/data_odometry_velodyne/dataset/sequences/00/output_map.pcd", *cloud) == -1) //* 保存点云到文件
//     {
//         PCL_ERROR("Couldn't write file\n");
//     }
//     std::cout << "Saved " << cloud->points.size() << " data points to output_map.pcd." << std::endl;
//     while(1);
//     std::cout<<"merge over!"<<std::endl;
    
//     std::vector<int> sem_labels,inst_labels;
//     for(int i = 0;i<cloud_label.size();i++)
//     {
//         sem_labels.push_back(cloud_label[i] & 0xFFFF);
//         inst_labels.push_back(cloud_label[i] >> 16);
//     }
//     // Eigen::VectorXi sem_labels = (labels.array() & 0xFFFF).matrix(); // 提取语义标签
//     // Eigen::VectorXi inst_labels = (labels.array() >> 16).matrix(); // 提取实例ID

//     // 验证重新组合的标签是否与原始标签相同
//     for (int i = 0; i < cloud_label.size(); ++i) {
//         assert((sem_labels[i] + (inst_labels[i] << 16)) == cloud_label[i]);
//     }

//     //重新映射sem_labels
//     for (int i = 0; i < sem_labels.size(); ++i) {
//         sem_labels[i] = remap_lut[sem_labels[i]];
//     }

//     std::set<int> sem_label_set_1(sem_labels.data(), sem_labels.data() + sem_labels.size());
//     std::vector<int> sem_label_set(sem_label_set_1.begin(), sem_label_set_1.end());
//     std::sort(sem_label_set.begin(), sem_label_set.end());

//     // auto gen_mid = std::chrono::high_resolution_clock::now();
//     // auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(gen_mid - gen_start);

//     // std::cout << "gen_mid:" <<duration1.count()<<std::endl;
//     //开始聚类
//     cluster.clear();
//     inst_id = 0;

//     for(int id_i=0;id_i<sem_label_set.size();id_i++)
//     {
//         std::cout<<"cluster:"<<sem_label_set[id_i]<<std::endl;
//         int label_i = sem_label_set[id_i];
//         std::vector<int> indice ;
//         for(int i = 0;i < sem_labels.size();i++)if(sem_label_set[id_i]==sem_labels[i])indice.push_back(i);
//       //  std::cout << "gen_labels 31" << std::endl;
//         std::vector<Eigen::Vector4f> sem_cluster;

//         for (auto i : indice)
//         {
//             Eigen::Vector4f row = cloud_map[i];
//             sem_cluster.push_back(row);
//         }

//         //std::cout << "gen_labels 32" << std::endl;
//         std::vector<int> tmp_inst_label;
//         for (int i = 0; i < indice.size(); ++i)tmp_inst_label.push_back(inst_labels[indice[i]]);
//         std::set<int> tmp_inst_label_set (tmp_inst_label.data(), tmp_inst_label.data() + tmp_inst_label.size());
//         std::vector<int> tmp_inst_set(tmp_inst_label_set.begin(), tmp_inst_label_set.end());
//         std::sort(tmp_inst_set.begin(), tmp_inst_set.end());

//         // std::cout << "erase cluster" << std::endl;
//         // std::vector<int> dete_indice =  indice;
//         // std::sort(dete_indice.begin(), dete_indice.end(), std::greater<int>());
//         // for (int index : dete_indice) {
//         //     std::cout<<index<<std::endl;
//         //     sem_labels.erase(sem_labels.begin() + index);
//         //     cloud_map.erase(cloud_map.begin() + index);
//         //     inst_labels.erase(inst_labels.begin() + index);
//         // }
//         // dete_indice.clear();

//         //std::cout << "gen_labels 4" << std::endl;
//         if (label_i == 9 || label_i == 10) 
//         {
//             //std::cout << "gen_labels 41" << std::endl;
//             std::vector<Eigen::VectorXf> inst_cluster;
//            // std::cout << "gen_labels 411" << std::endl;
//             // 扩展并填充 inst_cluster
//             for (const auto& vec : sem_cluster) {
//                 Eigen::VectorXf extended_vec(6);
//                 //std::cout<<"sem:"<<vec<<std::endl;
//                 extended_vec << vec, static_cast<float>(label_i), static_cast<float>(inst_id);
//                 inst_cluster.push_back(extended_vec);
//             }
//             //std::cout << "gen_labels 412" << std::endl;
//             inst_id += 1;  // 更新实例ID
//             cluster.insert(cluster.end(), inst_cluster.begin(), inst_cluster.end());
//             continue;
//         }
//         else if(label_i == 0 || label_i == 2 || label_i == 3 || label_i == 6 || label_i == 7 || label_i == 8)continue;
//         else if(tmp_inst_set.size() > 1 || (tmp_inst_set.size()==1 && tmp_inst_set[0] != 0))
//         {
//             //std::cout << "gen_labels 42" << std::endl;
//             for(int id_j = 0;id_j<tmp_inst_set.size();id_j++)
//             {
//                 int label_j = tmp_inst_set[id_j];
//                 std::vector<int> point_index ;
//                 for(int i = 0;i < tmp_inst_label.size();i++)if(tmp_inst_set[id_j]==tmp_inst_label[i])point_index.push_back(i);
//                 if(point_index.size()<=20)continue;

//                 std::vector<Eigen::VectorXf> inst_cluster;
//                 // 扩展并填充 inst_cluster
//                 for (auto i : point_index) {
//                     Eigen::VectorXf extended_vec(6);
//                     extended_vec << sem_cluster[i], static_cast<float>(label_i), static_cast<float>(inst_id);
//                     inst_cluster.push_back(extended_vec);
//                 }
//                 inst_id += 1;  // 更新实例ID
//                 cluster.insert(cluster.end(), inst_cluster.begin(), inst_cluster.end());
//                 continue;
//             }
//         }
//         else {
//             //std::cout << "gen_labels 3" << std::endl;
//             float cluster_tolerance = 0.2;  // 默认值
//             int min_size = 100;             // 默认值

//             switch (label_i) {
//                 case 1: case 4: case 5: case 14:
//                     cluster_tolerance = 0.5;
//                     break;
//                 case 11: case 12: case 13: case 15: case 17:
//                     cluster_tolerance = 2;
//                     if (label_i != 15) {
//                         min_size = 300;
//                     } else {
//                         min_size = 200;  // 特殊情况为vegetation
//                     }
//                     break;
//                 case 16: case 19:
//                     min_size = 50;
//                     break;
//                 default:
//                     // 默认值已经设置
//                     break;
//             }

//             auto gen_mid_1 = std::chrono::high_resolution_clock::now();


//             pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
//             for (const auto& point : sem_cluster) {
//                 cloud->push_back(pcl::PointXYZ(point[0], point[1], point[2]));
//             }
//             std::vector<pcl::PointIndices> cluster_indices;
//             cluster_indices = FEC(cloud, min_size, cluster_tolerance, 200000);
//             // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
//             // tree->setInputCloud(cloud);
            
//             // pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
//             // ec.setClusterTolerance(cluster_tolerance);
//             // ec.setMinClusterSize(min_size);
//             // ec.setMaxClusterSize(50000);
//             // ec.setSearchMethod(tree);
//             // ec.setInputCloud(cloud);

            
//             // ec.extract(cluster_indices);

//             // auto gen_mid_2 = std::chrono::high_resolution_clock::now();
//             // auto duration_mod = std::chrono::duration_cast<std::chrono::milliseconds>(gen_mid_2 - gen_mid_1);
//             // std::cout << "gen_mid_2:" <<duration_mod.count()<<std::endl;

//             for (size_t j = 0; j < cluster_indices.size(); j++) {
//                 auto indices = cluster_indices[j].indices;  // 使用正确的indices成员
//                 std::vector<Eigen::VectorXf> inst_cluster;
//                 // 扩展并填充 inst_cluster
//                 for (auto i : indices) {
//                     Eigen::VectorXf extended_vec(6);
//                     extended_vec << sem_cluster[i], static_cast<float>(label_i), static_cast<float>(inst_id);
//                     inst_cluster.push_back(extended_vec);
//                 }
//                 inst_id += 1;  // 更新实例ID
//                 cluster.insert(cluster.end(), inst_cluster.begin(), inst_cluster.end());
//             }
//             //std::cout << "gen_labels 5" << std::endl;
//         }

//     }
//     std::cout<<"cluster over!"<<std::endl;
//     std::vector<Eigen::VectorXf> scan = cluster;
//     cluster.clear();
//     std::vector<int> inst(scan.size());
//     std::transform(scan.begin(), scan.end(), inst.begin(), [](const Eigen::VectorXf& v) { return v[v.size() - 1]; });

//     std::set<int> inst_label_set1(inst.begin(), inst.end()); // 去重并排序
//     std::vector<int> inst_label_set(inst_label_set1.begin(), inst_label_set1.end()); // 如果需要有序vector

//     std::vector<int> nodes;
//     std::vector<Eigen::Vector2d> edges;
//     std::vector<float> weights;
//     std::vector<std::vector<Eigen::Vector3f>> cluster;
//     std::vector<Eigen::Vector3f> centers;

//     for(int id_i = 0;id_i < inst_label_set.size();id_i++)
//     {
//         std::vector<int> index ;
//         for(int i=0;i<inst.size();i++)if(inst_label_set[id_i] == inst[i])index.push_back(i);

//         std::vector<Eigen::VectorXf> inst_cluster ;
//         for(auto i : index)inst_cluster.push_back(scan[i]);
        
//         std::vector<int> sem_label_i(inst_cluster.size());
//         std::transform(inst_cluster.begin(), inst_cluster.end(), sem_label_i.begin(), [](const Eigen::VectorXf& v) { return v[v.size() - 2]; });
//         std::set<int> sem_label_set1(sem_label_i.begin(), sem_label_i.end()); // 去重并排序
//         std::vector<int> sem_label(sem_label_set1.begin(), sem_label_set1.end()); // 如果需要有序vector

//         assert(sem_label.size()==1);

//         if(node_map.find(sem_label[0]) != node_map.end())
//         {
//             Eigen::Vector3f center_now(0, 0, 0);
//             std::vector<Eigen::Vector3f> cluster_in;
//             for (const auto& vec : inst_cluster) {
//                 Eigen::Vector3f point = vec.head<3>(); // 取前三个元素
//                 center_now += vec.head<3>();
//                 cluster_in.push_back(point);
//             }
//             cluster.push_back(cluster_in);
//             center_now /= inst_cluster.size();
//             centers.push_back(center_now);
//             int node_label = node_map[sem_label[0]];
//             nodes.push_back(node_label);
//         }
//         else if(sem_label[0] == 9 || sem_label[0] == 10)continue;
//         else {
//             std::cout<<"wrong semantic label: "<<sem_label[0]<<std::endl;
//             continue;
//         }

//         // int  dist_thresh = 5;

//         // for (int i = 0; i < cluster.size() - 1; ++i) {
//         //     for (int j = i + 1; j < cluster.size(); ++j) {
//         //         // for (const auto& mat : cluster[i]) {
//         //         //     std::cout << "pco: " << mat.transpose() << std::endl;
//         //         // }

//         //         //std::cout<<"pco:"<<cluster[i]<<std::endl;
//         //         const auto& pc_i = cluster[i];
//         //         const auto& pc_j = cluster[j];

//         //         Eigen::Vector3f center_i(0, 0, 0);
//         //         Eigen::Vector3f center_j(0, 0, 0);
//         //         for (const auto& vec : pc_i) center_i += vec.head<3>();
//         //         center_i/=pc_i.size();
//         //         for (const auto& vec : pc_j) center_j += vec.head<3>();
//         //         center_j/=pc_j.size();


//         //         Eigen::Vector3f center = (center_i + center_j) / 2.0; // 计算中心点

//         //         // 找到最接近中心点的点
//         //         float min_dis_i = (center - center_i).norm();
//         //         float min_dis_j = (center - center_j).norm();
//         //         float min_dis = std::min(min_dis_i, min_dis_j);

//         //         if (min_dis <= dist_thresh) {
//         //             edges.push_back({i, j}); // 添加边
//         //             float weight = 1 - min_dis / dist_thresh;
//         //             weights.push_back(weight); // 添加权重
//         //         }

//         //     }
//         // }
 
//     }
//     Graph_map graph_f(nodes,edges,weights,centers,cloud_poses);

//     auto jsonGraph = graph_f.toJSON();
//     // size_t last_slash_idx = scan_name.find_last_of("\\/");
//     // std::string raw_name = scan_name.substr(last_slash_idx + 1);
//     // size_t dot_idx = raw_name.find_last_of('.');
//     // std::string base_name = raw_name.substr(0, dot_idx);
//     std::string file_name = "/media/beabbit/T5 EVO/data_odometry_velodyne/dataset/" +seq_path+".json";  // 使用适当的文件名
//     std::ofstream file1(file_name);
//    // std::cout<<"file:"<<file_name<<std::endl;
//     file1 << jsonGraph;  // 美化输出
// }



std::vector<Eigen::VectorXf> gen_labels(std::string scan_name,std::string label_name,std::string label_output_dir)
{
    

    // auto gen_start = std::chrono::high_resolution_clock::now();
    // 打开二进制点云文件
    std::ifstream scan_file(scan_name, std::ios::binary);
    // 读取数据到vector中
    //std::vector<float> scan_data((std::istreambuf_iterator<char>(scan_file)), std::istreambuf_iterator<char>());
    scan_file.seekg(0, std::ios::end);
    size_t size = scan_file.tellg();
    scan_file.seekg(0, std::ios::beg);

    // 根据文件大小和浮点数大小计算点的数量
    size_t num_points = size / sizeof(float);

    // 读取浮点数到vector中
    std::vector<float> scan_data(num_points);
    scan_file.read(reinterpret_cast<char*>(scan_data.data()), size);

    // 将数据映射到Eigen矩阵
    Eigen::Map<Eigen::MatrixXf> points(scan_data.data(), 4, num_points / 4);

    // 使用Eigen的Map将数据映射为矩阵，方便处理。这里假设每个点有4个浮点数（x, y, z, intensity）
   // Eigen::Map<Eigen::MatrixXf> points(scan_data.data(), 4, scan_data.size() / 4);

    // std::cout << " point " << points<< std::endl;
  //  std::cout << " gen_labels 1 " << std::endl;
    // 计算每个点的角度
    Eigen::VectorXf angles = points.row(1).binaryExpr(points.row(0), static_cast<float(*)(float, float)>(std::atan2));

    //std::cout<<"angle="<<points.col(1)<<std::endl;
    //while(1);
    // 类似地打开并读取标签文件
    std::ifstream label_file(label_name, std::ios::binary);
    std::vector<uint32_t> label_data;

    label_file.seekg(0, std::ios::end);
    size_t size_la = label_file.tellg();
    label_file.seekg(0, std::ios::beg);

    // 根据文件大小调整vector大小（每个uint32_t占4字节）
    label_data.resize(size_la / sizeof(uint32_t));

    // 直接读取到vector的内存中
    label_file.read(reinterpret_cast<char*>(label_data.data()), size_la);

    // 将标签数据映射为Eigen整数向量
    Eigen::Map<Eigen::VectorXi> labels(reinterpret_cast<int*>(label_data.data()), label_data.size());
    // std::cout << " label " << labels<< std::endl;
    // 之后可以进行其他处理...
   // std::cout << "gen_labels 2" << std::endl;
    assert(points.cols()==labels.size());
    
    std::vector<int> sem_labels,inst_labels;
    for(int i = 0;i<labels.size();i++)
    {
        sem_labels.push_back(labels[i] & 0xFFFF);
        inst_labels.push_back(labels[i] >> 16);
    }
    // Eigen::VectorXi sem_labels = (labels.array() & 0xFFFF).matrix(); // 提取语义标签
    // Eigen::VectorXi inst_labels = (labels.array() >> 16).matrix(); // 提取实例ID

    // 验证重新组合的标签是否与原始标签相同
    for (int i = 0; i < labels.size(); ++i) {
        assert((sem_labels[i] + (inst_labels[i] << 16)) == labels[i]);
    }

    //重新映射sem_labels
    // for (int i = 0; i < sem_labels.size(); ++i) {
    //     sem_labels[i] = remap_lut[sem_labels[i]];
    // }

    std::set<int> sem_label_set_1(sem_labels.data(), sem_labels.data() + sem_labels.size());
    std::vector<int> sem_label_set(sem_label_set_1.begin(), sem_label_set_1.end());
    std::sort(sem_label_set.begin(), sem_label_set.end());

    // auto gen_mid = std::chrono::high_resolution_clock::now();
    // auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(gen_mid - gen_start);

    // std::cout << "gen_mid:" <<duration1.count()<<std::endl;
    //开始聚类
    cluster.clear();
    inst_id = 0;
/*
    std::vector<std::thread> threads;
    int threads_num = std::min(static_cast<int>(sem_label_set.size()), 6);
    for (int threads_i = 0; threads_i < threads_num; ++threads_i) {
        int start_index = (sem_label_set.size() * threads_i) / threads_num;
        int end_index = (sem_label_set.size() * (threads_i + 1)) / threads_num;

        // 创建子集合
        std::vector<int> suzu(sem_label_set.begin() + start_index, sem_label_set.begin() + end_index);

        // 创建线程
        threads.emplace_back(process_label, std::ref(suzu), std::ref(sem_label_set), std::ref(sem_labels), std::ref(points), std::ref(inst_labels));
    }

    for (auto& thread : threads) {
        thread.join();
    }
*/
    for(int id_i=0;id_i<sem_label_set.size();id_i++)
    {
        int label_i = sem_label_set[id_i];
        std::vector<int> indice ;
        for(int i = 0;i < sem_labels.size();i++)if(sem_label_set[id_i]==sem_labels[i])indice.push_back(i);
      //  std::cout << "gen_labels 31" << std::endl;
        std::vector<Eigen::Vector4f> sem_cluster;

        for (auto i : indice)
        {
            Eigen::Vector4f row = points.col(i);
            sem_cluster.push_back(row);
        }
        //std::cout << "gen_labels 32" << std::endl;
        std::vector<int> tmp_inst_label;
        for (int i = 0; i < indice.size(); ++i)tmp_inst_label.push_back(inst_labels[indice[i]]);
        std::set<int> tmp_inst_label_set (tmp_inst_label.data(), tmp_inst_label.data() + tmp_inst_label.size());
        std::vector<int> tmp_inst_set(tmp_inst_label_set.begin(), tmp_inst_label_set.end());
        std::sort(tmp_inst_set.begin(), tmp_inst_set.end());

        //std::cout << "gen_labels 4" << std::endl;
        if (label_i == 9 || label_i == 10) 
        {
            //std::cout << "gen_labels 41" << std::endl;
            std::vector<Eigen::VectorXf> inst_cluster;
           // std::cout << "gen_labels 411" << std::endl;
            // 扩展并填充 inst_cluster
            for (const auto& vec : sem_cluster) {
                Eigen::VectorXf extended_vec(6);
                //std::cout<<"sem:"<<vec<<std::endl;
                extended_vec << vec, static_cast<float>(label_i), static_cast<float>(inst_id);
                inst_cluster.push_back(extended_vec);
            }
            //std::cout << "gen_labels 412" << std::endl;
            inst_id += 1;  // 更新实例ID
            cluster.insert(cluster.end(), inst_cluster.begin(), inst_cluster.end());
            continue;
        }
        else if(label_i == 0 || label_i == 1 || label_i == 2 || label_i == 3 || label_i == 6 || label_i == 7 || label_i == 8 || label_i == 14)continue;
        else if(tmp_inst_set.size() > 1 || (tmp_inst_set.size()==1 && tmp_inst_set[0] != 0))
        {
            //std::cout << "gen_labels 42" << std::endl;
            for(int id_j = 0;id_j<tmp_inst_set.size();id_j++)
            {
                int label_j = tmp_inst_set[id_j];
                std::vector<int> point_index ;
                for(int i = 0;i < tmp_inst_label.size();i++)if(tmp_inst_set[id_j]==tmp_inst_label[i])point_index.push_back(i);
                if(point_index.size()<=20)continue;

                std::vector<Eigen::VectorXf> inst_cluster;
                // 扩展并填充 inst_cluster
                for (auto i : point_index) {
                    Eigen::VectorXf extended_vec(6);
                    extended_vec << sem_cluster[i], static_cast<float>(label_i), static_cast<float>(inst_id);
                    inst_cluster.push_back(extended_vec);
                }
                inst_id += 1;  // 更新实例ID
                cluster.insert(cluster.end(), inst_cluster.begin(), inst_cluster.end());
                continue;
            }
        }
        else {
            //std::cout << "gen_labels 3" << std::endl;
            float cluster_tolerance = 0.5;  // 默认值
            int min_size = 100;             // 默认值
            int DCVC_min = 300;
            switch (label_i) {
                case 1: case 4: case 5: case 13:
                    cluster_tolerance = 0.5;
                    // DCVC_min = 100;
                    break;
                case 10: case 11: case 12: case 14: case 16:
                    cluster_tolerance = 0.8;
                    if (label_i != 14) {
                        min_size = 300;
                    } else {
                        min_size = 200;  // 特殊情况为vegetation
                    }
                    break;
                case 17: case 18: case 15:
                    min_size = 15;
                    DCVC_min = 5;
                    break;
                
                default:
                    // 默认值已经设置
                    break;
            }

            auto gen_mid_1 = std::chrono::high_resolution_clock::now();


            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
            for (const auto& point : sem_cluster) {
                cloud->push_back(pcl::PointXYZ(point[0], point[1], point[2]));
            }
            
            // std::vector<pcl::PointIndices> cluster_indices;
            // cluster_indices = FEC(cloud, min_size, cluster_tolerance, 50000);
            
            // std::cout<<"start!!"<<std::endl;
            clusterManager cluster_node ;
            cluster_node.params_.clusterTolerance = cluster_tolerance;
            cluster_node.params_.minClusterSize = min_size;
            cluster_node.params_.maxClusterSize = 100000;

            cluster_node.params_.startR= 0.35    ;
            cluster_node.params_.deltaR= 0.0004  ;
            cluster_node.params_.deltaP= 1.2   ;  
            cluster_node.params_.deltaA= 1.2   ;    
            cluster_node.params_.minSeg= DCVC_min  ;    

            // cluster_node.DCVCParam.startR= 0.35    
            // cluster_node.DCVCParam.deltaR= 0.0004  
            // cluster_node.DCVCParam.deltaP= 1.2     
            // cluster_node.DCVCParam.deltaA= 1.2       
            // cluster_node.DCVCParam.minSeg= 300     
            // std::cout<<"start!!2"<<std::endl;
            cluster_node.segmentPointCloud(cloud);
            // std::cout<<"start!!3"<<std::endl;
            for (const auto& sem_cluster :cluster_node.clusters_) {
                // auto indices = cluster_indices[j].indices;  // 使用正确的indices成员
                std::vector<Eigen::VectorXf> inst_cluster;
                // 扩展并填充 inst_cluster
                for (const auto& point : sem_cluster->points) { 
                    Eigen::VectorXf extended_vec(6);
                    extended_vec << point.x,point.y,point.z,1, static_cast<float>(label_i), static_cast<float>(inst_id);
                    inst_cluster.push_back(extended_vec);
                }
                inst_id += 1;  // 更新实例ID
                cluster.insert(cluster.end(), inst_cluster.begin(), inst_cluster.end());
            }
            // std::cout<<"start!!4"<<std::endl;
            // cluster_indices = cluster_node.clusters_;

            // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
            // tree->setInputCloud(cloud);
            
            // pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
            // ec.setClusterTolerance(cluster_tolerance);
            // ec.setMinClusterSize(min_size);
            // ec.setMaxClusterSize(50000);
            // ec.setSearchMethod(tree);
            // ec.setInputCloud(cloud);

            
            // ec.extract(cluster_indices);

            // auto gen_mid_2 = std::chrono::high_resolution_clock::now();
            // auto duration_mod = std::chrono::duration_cast<std::chrono::milliseconds>(gen_mid_2 - gen_mid_1);
            // std::cout << "gen_mid_2:" <<duration_mod.count()<<std::endl;

            // for (size_t j = 0; j < cluster_indices.size(); j++) {
            //     auto indices = cluster_indices[j].indices;  // 使用正确的indices成员
            //     std::vector<Eigen::VectorXf> inst_cluster;
            //     // 扩展并填充 inst_cluster
            //     for (auto i : indices) { 
            //         Eigen::VectorXf extended_vec(6);
            //         extended_vec << sem_cluster[i], static_cast<float>(label_i), static_cast<float>(inst_id);
            //         inst_cluster.push_back(extended_vec);
            //     }
            //     inst_id += 1;  // 更新实例ID
            //     cluster.insert(cluster.end(), inst_cluster.begin(), inst_cluster.end());
            // }
            // std::cout << "gen_labels 5" << std::endl;
        }
    }

    // for (const auto& mat : cluster) {
    //     std::cout << mat.transpose() << std::endl; // 假设mat是行向量
    // }
    // while(1);
    // auto gen_end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start);
    // std::cout << "gen_end:" <<duration.count()<<std::endl;
    return cluster;
}

void gen_graphs(std::string scan_name,std::vector<Eigen::VectorXf> scan,std::string graph_output_dir,std::vector<float> poseRow)
{    
    std::vector<int> inst(scan.size());
    std::transform(scan.begin(), scan.end(), inst.begin(), [](const Eigen::VectorXf& v) { return v[v.size() - 1]; });

    std::set<int> inst_label_set1(inst.begin(), inst.end()); // 去重并排序
    std::vector<int> inst_label_set(inst_label_set1.begin(), inst_label_set1.end()); // 如果需要有序vector

    std::vector<int> nodes;
    std::vector<Eigen::Vector2d> edges;
    std::vector<float> weights;
    std::vector<float> volumes;
    std::vector<float> densitys;
    std::vector<std::vector<Eigen::Vector3f>> cluster;
    std::vector<Eigen::Vector3f> centers;
    // for (auto i :inst_label_set)std::cout<<i<<" ";
    // std::cout<<std::endl;
    
    for(int id_i = 0;id_i < inst_label_set.size();id_i++)
    {
        std::vector<int> index ;
        for(int i=0;i<inst.size();i++)if(inst_label_set[id_i] == inst[i])index.push_back(i);

        std::vector<Eigen::VectorXf> inst_cluster ;
        for(auto i : index)inst_cluster.push_back(scan[i]);
        
        std::vector<int> sem_label_i(inst_cluster.size());
        std::transform(inst_cluster.begin(), inst_cluster.end(), sem_label_i.begin(), [](const Eigen::VectorXf& v) { return v[v.size() - 2]; });
        std::set<int> sem_label_set1(sem_label_i.begin(), sem_label_i.end()); // 去重并排序
        std::vector<int> sem_label(sem_label_set1.begin(), sem_label_set1.end()); // 如果需要有序vector

        assert(sem_label.size()==1);

        if(node_map.find(sem_label[0]) != node_map.end())
        {
            Eigen::Vector3f center_now(0, 0, 0);
            std::vector<Eigen::Vector3f> cluster_in;
            for (const auto& vec : inst_cluster) {
                Eigen::Vector3f point = vec.head<3>(); // 取前三个元素
                center_now += vec.head<3>();
                cluster_in.push_back(point);
            }
            
            center_now /= inst_cluster.size();
            
            //去质心化

            float Cetner_Gass=0.0;
            size_t i = 0;
            for (const auto& vec : inst_cluster) {
                Eigen::Vector3f point = vec.head<3>(); // 取前三个元素
                Cetner_Gass += ((point - center_now).transpose().dot(point - center_now))/inst_cluster.size();
                i++;
            }


            // // 计算协方差矩阵
            // Eigen::Matrix3f cov = centered_points.transpose() * centered_points / float(inst_cluster.size());

            // // 计算特征值和特征向量
            // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(cov);
            // Eigen::Vector3f eigenvalues = eigen_solver.eigenvalues();
            // Eigen::Matrix3f eigenvectors = eigen_solver.eigenvectors();

            // // 按特征值从大到小排序
            // std::vector<int> indices(3);
            // std::iota(indices.begin(), indices.end(), 0);
            // std::sort(indices.begin(), indices.end(), [&eigenvalues](int i1, int i2) { return eigenvalues(i1) > eigenvalues(i2); });

            // Eigen::Matrix3f sorted_eigenvectors;
            // for (int i = 0; i < 3; ++i) {
            //     sorted_eigenvectors.col(i) = eigenvectors.col(indices[i]);
            // }

            // // 将点云投影到主成分轴上
            // Eigen::MatrixXf projected_points = centered_points * sorted_eigenvectors;

            // // 计算投影点的最小和最大值
            // Eigen::Vector3f min_bound = projected_points.colwise().minCoeff();
            // Eigen::Vector3f max_bound = projected_points.colwise().maxCoeff();

            // // 计算包围盒尺寸
            // Eigen::Vector3f lengths = max_bound - min_bound;

            // // 计算体积
            // float volume = lengths(0) * lengths(1) * lengths(2);

            // // 计算点数量
            // size_t num_points = inst_cluster.size();

            // // 计算密度
            // float density = num_points / volume;
            // volumes.push_back(volume);
            

            int node_label = node_map[sem_label[0]];
            if (node_label >=3 && node_label <= 12 ){
                nodes.push_back(node_label);
                centers.push_back(center_now);
                densitys.push_back(Cetner_Gass);
                cluster.push_back(cluster_in);
                }
        }
        else if(sem_label[0] == 9 || sem_label[0] == 10)continue;
        else {
            std::cout<<"wrong semantic label: "<<sem_label[0]<<std::endl;
            continue;
        }
    }
    int  dist_thresh = 5;

    for (int i = 0; i < cluster.size() - 1; ++i) {
        for (int j = i + 1; j < cluster.size(); ++j) {

            const auto& pc_i = cluster[i];
            const auto& pc_j = cluster[j];

            Eigen::Vector3f center_i=centers[i];
            Eigen::Vector3f center_j=centers[j];
            // for (const auto& vec : pc_i) center_i += vec.head<3>();
            // center_i/=pc_i.size();
            // for (const auto& vec : pc_j) center_j += vec.head<3>();
            // center_j/=pc_j.size();
            Eigen::Vector3f center = (center_i + center_j) / 2.0; // 计算中心点

            // 找到最接近中心点的点
            float min_dis_i = (center - center_i).norm();
            float min_dis_j = (center - center_j).norm();
            float min_dis = std::min(min_dis_i, min_dis_j);

            if (min_dis <= dist_thresh) {
                edges.push_back({i, j}); // 添加边
                float weight = 1 - min_dis / dist_thresh;
                weights.push_back(weight); // 添加权重
            }

        }
    }
 
    
    Graph graph_f(nodes,edges,weights,centers,poseRow,volumes,densitys);

    auto jsonGraph = graph_f.toJSON();
    size_t last_slash_idx = scan_name.find_last_of("\\/");
    std::string raw_name = scan_name.substr(last_slash_idx + 1);
    size_t dot_idx = raw_name.find_last_of('.');
    std::string base_name = raw_name.substr(0, dot_idx);
    std::string file_name = graph_output_dir + '/' +base_name+".json";  // 使用适当的文件名
    std::ofstream file(file_name);
   // std::cout<<"file:"<<file_name<<std::endl;
    file << jsonGraph;  // 美化输出

}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;


/*  my code in semantic scan match  */

    //param init
    /*
    dataset: "/media/beabbit/T5 EVO/data_odometry_velodyne/dataset/" 
    config : "config/semantic-kitti.yaml"
    output_label: "/media/beabbit/T5 EVO/data_odometry_velodyne/dataset/labels"
    output_graph: "/media/beabbit/T5 EVO/data_odometry_velodyne/dataset/graphs"
    label_topic: "/labeled_pc"
    graph_topic: "/graphs"
    */

    // // 定义平移和旋转参数
    // Eigen::Vector3d trans(1.7042, -0.021, 1.8047);
    // double roll = 0.0001, pitch = 0.0003, yaw = 179.6654;

    // // 创建从欧拉角到旋转矩阵的转换
    // Eigen::Matrix3d rot = Eigen::AngleAxisd(yaw * M_PI/180, Eigen::Vector3d::UnitZ())
    //                       * Eigen::AngleAxisd(pitch * M_PI/180, Eigen::Vector3d::UnitY())
    //                       * Eigen::AngleAxisd(roll * M_PI/180, Eigen::Vector3d::UnitX());

    // // 创建4x4的变换矩阵
    // Eigen::Matrix4d BASE2OUSTER = Eigen::Matrix4d::Identity();
    // BASE2OUSTER.block<3,3>(0, 0) = rot;
    // BASE2OUSTER.block<3,1>(0, 3) = trans;

    // // 输出结果
    // std::cout << "Transform Matrix: \n" << BASE2OUSTER << std::endl;
    // while(1);

    if(1)
    {   std::string dataset_path,data_config_path,output_label_path,output_graph_path,data_label_topic,data_graph_topic;
        nh.param<string>("data_process/dataset",dataset_path,"/media/beabbit/T5 EVO/data_odometry_velodyne/dataset/");
        nh.param<string>("data_process/config",data_config_path,"/media/beabbit/T5 EVO/data_odometry_velodyne/dataset/semantic-kitti.yaml");
        nh.param<string>("data_process/output_label",output_label_path,"/media/beabbit/T5 EVO/data_odometry_velodyne/EC/labels");
        nh.param<string>("data_process/output_graph",output_graph_path,"/media/beabbit/T5 EVO/data_odometry_velodyne/dataset/graphs_nomove_try");
        nh.param<string>("data_process/label_topic",data_label_topic,"/labeled_pc");
        nh.param<string>("data_process/graph_topic",data_graph_topic,"/graphs");
        
        YAML::Node kitti_config_file = YAML::LoadFile(data_config_path);
        YAML::Node color_map = kitti_config_file["color_map"];
        YAML::Node learning_map_inv_yaml = kitti_config_file["learning_map_inv"];
        std::vector<string> sequences = {"Riverside01"};//,"DCC01","kaist01","kaist02","kaist03"
        
        //初始化查找表
        for (const auto& pair : learning_map) {
            remap_lut[pair.first] = pair.second;
        }

        Eigen::MatrixXi colormaps(260,3);
        Eigen::VectorXi learning_map_inv(20);
        for(YAML::const_iterator it = learning_map_inv_yaml.begin(); it !=learning_map_inv_yaml.end();++it)
        {
            int key = it->first.as<int>();
            int value = it->second.as<int>();
            learning_map_inv(key)=value;
            // std::cout<<"key:"<<key<<" value:"<<value<<std::endl;
        }
        for(YAML::const_iterator it = color_map.begin(); it !=color_map.end();++it)
        {
            int key = it->first.as<int>();
            std::vector<int> rbg = it->second.as<std::vector<int>>();
            colormaps.row(key) = Eigen::Vector3i(rbg[0],rbg[1],rbg[2]);
        }

        for(auto seq : sequences){
            //init
            std::cout<<std::string(80,'*')<<std::endl;
            // std::stringstream seq_1;
            // seq_1 << std::setw(2)<< std::setfill('0')<<seq;
            // std::string seq_path = seq_1.str();
            std::cout<<"seq:"<<seq<<std::endl;
            // if (seq_path!="00")break;
            // std::string poses_dir = dataset_path +"poses/"+seq_path+".txt";
            // std::string poses_dir = "/media/beabbit/T5 EVO/bag/rosbag_/mulran/DCC03/sensor_data/pose.txt";
            // std::ifstream file(poses_dir); // 假设pose.txt是你的文件名
            // std::string line;

            
            //label out dir init
            std::string label_output_dir = output_label_path + seq +"/labels"+ '/' ;
            if (!boost::filesystem::exists(label_output_dir)) {
                boost::filesystem::create_directories(label_output_dir);
            }

            //graph out dir init
            std::string graph_output_dir = output_graph_path + seq +"/graph_DCVC_10w"+ '/' ;
            if (!boost::filesystem::exists(graph_output_dir)) {
                boost::filesystem::create_directories(graph_output_dir);
            }

            //does sequence folder exist?
            // std::string scan_paths = dataset_path + "sequences" + '/' + seq_path +'/' + "velodyne";
            std::string scan_paths = dataset_path + seq  + "/sensor_data/Ouster";
            if (boost::filesystem::is_directory(scan_paths)) {
                std::cout << "Sequence folder exists!" << std::endl;
            } else {
                std::cout << "Sequence folder doesn't exist! Exiting..." << std::endl;
                exit(1); // 退出程序
            }
            std::vector<std::string> scan_names;
            for (const auto& entry : boost::filesystem::recursive_directory_iterator(scan_paths)) {
                if (boost::filesystem::is_regular_file(entry)) {
                    scan_names.push_back(entry.path().string());
                }
            }
            // 对文件名进行排序
            std::sort(scan_names.begin(), scan_names.end());
            
            //does sequence folder exist?
            // std::string label_paths = dataset_path + "sequences" + '/' + seq_path +'/' + "labels";
            std::string label_paths = "/media/beabbit/T5 EVO/bag/rosbag_/mulran/spvnas/"+seq+"/sensor_data";
            if (boost::filesystem::is_directory(label_paths)) {
                std::cout << "Labels folder exists!" << std::endl;
            } else {
                std::cout << "Labels folder doesn't exist! Exiting..." << std::endl;
                exit(1); // 退出程序
            }
            std::vector<std::string> label_names;
            for (const auto& entry : boost::filesystem::recursive_directory_iterator(label_paths)) {
                if (boost::filesystem::is_regular_file(entry)) {
                    label_names.push_back(entry.path().string());
                }
            }
            // 对文件名进行排序
            std::sort(label_names.begin(), label_names.end());
            std::cout << scan_paths <<" "<< std::endl;
            assert(label_names.size()==scan_names.size());
            std::ofstream time_file("/home/beabbit/study_space/SGPR/SG_PR-master/data_process/ctime.txt");


            // gen_map(scan_names,label_names,poses_dir,label_output_dir,seq_path);

            for(int i=0;i < scan_names.size();i++)
            {
                std::string poses_dir = "/media/beabbit/T5 EVO/bag/rosbag_/mulran/"+seq+"/sensor_data/pose.txt";
                std::ifstream file(poses_dir); // 假设pose.txt是你的文件名
                std::string line;
                //std::cout<<"start;"<<std::endl;
                auto start = std::chrono::high_resolution_clock::now();
                long long dt_tb = 0;
                float value;
                std::string last_iss;
                std::string now_iss;
                std::vector<float> poseRow;
                long long timestamp_now;
                std::string base_name ;
                while(getline(file, line))
                {
                    // std::cout<<line<<std::endl;
                    std::istringstream iss(line);
                    bool fir = false;
                    while (iss >> timestamp_now) {
                        break;
                    }
                    size_t last_slash_idx = scan_names[i].find_last_of("\\/");
                    std::string raw_name = scan_names[i].substr(last_slash_idx + 1);
                    size_t dot_idx = raw_name.find_last_of('.');
                    base_name = raw_name.substr(0, dot_idx);
                    // std::cout<<timestamp_now<<" "<<std::stoll(base_name)<<" "<<std::abs(timestamp_now-std::stoll(base_name))<<std::endl;
                    if(timestamp_now-std::stoll(base_name)>0)
                    {
                        if(std::abs(timestamp_now-std::stoll(base_name)) > dt_tb) now_iss = last_iss;
                        else now_iss = line;
                        // std::cout<<now_iss<<std::endl;
                        break;
                    }
                    dt_tb = std::abs(timestamp_now-std::stoll(base_name));
                    last_iss = line;
                }
                std::istringstream iss(now_iss);
                std::string new_iss;
                char douh;
                iss >> dt_tb >>douh>>new_iss;
                // while (iss >> dt_tb) {
                //     break;
                // }
                // std::cout<<"dt_tb:"<<dt_tb<< " " <<new_iss<<std::endl;
                
                for (int id_c = 0;id_c<new_iss.size();id_c++)
                {
                    if(new_iss[id_c] == ',')new_iss[id_c] = ' ';
                }
                // std::cout<<new_iss<<std::endl;
                std::istringstream is_iss(new_iss);
                while (is_iss >> value) {
                    // if(!fir){
                    //     fir = true;
                    // }
                    // else 
                    // std::cout<<value<<std::endl;
                    poseRow.push_back(value);
                }
                if (poseRow.size()==0)
                {
                    std::cout<<timestamp_now<<" "<<std::stoll(base_name)<<" "<<std::abs(timestamp_now-std::stoll(base_name))<<std::endl;
                    break;
                }
                if (seq == "kaist01" || seq =="kaist02" || seq =="kaist03")
                {
                    poseRow[3] = poseRow[3]-353050;//kaist
                    poseRow[7] = poseRow[7]-4026791;
                    poseRow[11] = poseRow[11]-19;
                }
                else if(seq == "DCC01" || seq =="DCC02" || seq =="DCC03"){
                    poseRow[3] = poseRow[3]-355630;//dcc
                    poseRow[7] = poseRow[7]-402670;
                    poseRow[11] = poseRow[11]-19;
                }
                else if(seq == "Riverside02" || seq =="Riverside01"){
                    poseRow[3] = poseRow[3]-353610;//dcc
                    poseRow[7] = poseRow[7]-4025986;
                    poseRow[11] = poseRow[11]-19;
                }
                // std::cout<<poseRow[3]<<" "<< poseRow[7]<<" "<<poseRow[11]<<std::endl;

                // std::cout<<poseRow.size()<<std::endl;
                // break;
                //下面的变量名称自拟      
                // Eigen::Quaterniond q_odom_curr_tmp;//声明一个Eigen类的四元数
                // //此处进行赋值，使用其他语句以及合理的常数也可
                // q_odom_curr_tmp.x() = poseRow[4];
                // q_odom_curr_tmp.y() = poseRow[5];
                // q_odom_curr_tmp.z() = poseRow[6];
                // q_odom_curr_tmp.w() =poseRow[7];
                // Eigen::Matrix3d R_odom_curr_tmp;//声明一个Eigen类的3*3的旋转矩阵
                // float lx,ly,lz;
                // lx = poseRow[1];
                // ly = poseRow[2];
                // lz = poseRow[3];
                // //四元数转为旋转矩阵--先归一化再转为旋转矩阵
                // R_odom_curr_tmp= q_odom_curr_tmp.normalized().toRotationMatrix();  
                // poseRow.clear();
                // poseRow.push_back(R_odom_curr_tmp(0,0));poseRow.push_back(R_odom_curr_tmp(0,1));poseRow.push_back(R_odom_curr_tmp(0,2));poseRow.push_back(lx);
                // poseRow.push_back(R_odom_curr_tmp(1,0));poseRow.push_back(R_odom_curr_tmp(1,1));poseRow.push_back(R_odom_curr_tmp(1,2));poseRow.push_back(ly);
                // poseRow.push_back(R_odom_curr_tmp(2,0));poseRow.push_back(R_odom_curr_tmp(2,1));poseRow.push_back(R_odom_curr_tmp(2,2));poseRow.push_back(lz);

                std::vector<Eigen::VectorXf>  clusters = gen_labels(scan_names[i],label_names[i],label_output_dir);
                // auto stop1 = std::chrono::high_resolution_clock::now();
                // auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start);
                //std::cout<<"time1;"<<duration1.count()<<"size:"<<scan_names.size()<<" i="<<i<<std::endl;
                gen_graphs(scan_names[i],clusters,graph_output_dir,poseRow);
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                std::cout<<"time:"<<duration.count()<<" size:"<<scan_names.size()<<" i="<<i<<" seq:"<<seq<<std::endl;
                time_file << duration.count() <<std::endl;

            }
            //while(1);

        }   

    // std::vector<string> color_map = kitti_config_file["color_map"].as<std::vector<string>>();

        // for(auto s : color_map ){
        //     std::cout<<s<<" ";
        // }
        // for(int i=0;i<learning_map_inv.rows();++i)std::cout<<"Key:"<<i<<" learning:"<<learning_map_inv(i)<<std::endl; 
        return 0;
        std::cout<<endl;while(1);}

    return 0;
}
