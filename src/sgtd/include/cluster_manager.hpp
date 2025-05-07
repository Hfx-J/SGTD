#ifndef CLUSTER_MANAGER_H
#define CLUSTER_MANAGER_H
/*2024 07 23*/
/*移植DCVC算法*/
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
/*点云体素化*/
template <typename T>
void voxelize(const boost::shared_ptr<pcl::PointCloud<T>> srcPtr,
              boost::shared_ptr<pcl::PointCloud<T>> dstPtr,
              double voxelSize) {
    static pcl::VoxelGrid<T> voxel_filter;
    voxel_filter.setInputCloud(srcPtr);
    voxel_filter.setLeafSize(voxelSize, voxelSize, voxelSize);
    voxel_filter.filter(*dstPtr);
}

template <typename T>
void voxelize(pcl::PointCloud<T>& src,
              boost::shared_ptr<pcl::PointCloud<T>> dstPtr,
              double voxelSize) {
    static pcl::VoxelGrid<T> voxel_filter;
    voxel_filter.setInputCloud(src);
    voxel_filter.setLeafSize(voxelSize, voxelSize, voxelSize);
    voxel_filter.filter(*dstPtr);
}

typedef struct InstanceClusters {
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr centroids_;
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>&
            covariances_;
} InstanceClusters;


/*类开始的地方*/
class clusterManager {
    public:
    pcl::PointCloud<pcl::PointXYZ>::Ptr sem_cloud_in; //输入点云？
    clusterManager() { //初始化
        tree_.reset(new pcl::search::KdTree<pcl::PointXYZ>);
        selected_points_.reset(new pcl::PointCloud<pcl::PointXYZ>);
        // selected_src_points_.reset(new pcl::PointCloud<PointType>);
        // selected_tgt_points_.reset(new pcl::PointCloud<PointType>);
        // src_descriptors_.reset(new pcl::PointCloud<pcl::FPFHSignature33>());
        // /tgt_descriptors_.reset(new pcl::PointCloud<pcl::FPFHSignature33>());
    }

    /** \brief Empty destructor */
    ~clusterManager() {
        tree_.reset(new pcl::search::KdTree<pcl::PointXYZ>);
        selected_points_.reset(new pcl::PointCloud<pcl::PointXYZ>);
        // selected_src_points_.reset(new pcl::PointCloud<PointType>);
        // selected_tgt_points_.reset(new pcl::PointCloud<PointType>);
        // src_descriptors_.reset(new pcl::PointCloud<pcl::FPFHSignature33>());
        // tgt_descriptors_.reset(new pcl::PointCloud<pcl::FPFHSignature33>());
    }

    //参数设置
    struct DCVCParam {
        double startR = 0.0;
        double deltaR = 0.0;
        double deltaP = 0.0;
        double deltaA = 0.0;
        int minSeg = 0;
    };

    struct ClusterParams {
        // semantic class
        int semanticLabel = 1; // 考虑去除这部分，直接把区分好的点云输入进来
        // DBScan clustering related params
        double clusterTolerance = 0.5;
        int minClusterSize = 20;
        int maxClusterSize = 2000;
        // DCVD params
        double startR = 0.0;
        double deltaR = 0.0;
        double deltaP = 0.0;
        double deltaA = 0.0;
        int minSeg = 0;
    };

    void reset(ClusterParams params) {
        params_ = params;
        // reg_method_ = RegularizationMethod::NONE;
    }

    //选择语义点，将不是该语义信息的点删除，考虑去除
    bool selectSemanticPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr input_sem_cloud) {
        sem_cloud_in = input_sem_cloud;

        for (int i = 0; i < sem_cloud_in->points.size(); i++) {
            // if (sem_cloud_in->points[i].label == params_.semanticLabel) {
                pcl::PointXYZ tmpPt;
                tmpPt.x = sem_cloud_in->points[i].x;
                tmpPt.y = sem_cloud_in->points[i].y;
                tmpPt.z = sem_cloud_in->points[i].z;
                selected_points_->points.push_back(tmpPt);
            // }
        }

        if (selected_points_->points.size() < 0) return false;

        return true;
    }
    //重构函数，作用同上
    // bool selectSemanticPoints(
    //         pcl::PointCloud<pcl::PointXYZ>::Ptr input_src_sem_cloud,
    //         pcl::PointCloud<pcl::PointXYZ>::Ptr input_tgt_sem_cloud) {
    //     for (int i = 0; i < input_src_sem_cloud->points.size(); i++) {
    //         if (input_src_sem_cloud->points[i].label == params_.semanticLabel) {
    //             PointType tmpPt;
    //             tmpPt.x = input_src_sem_cloud->points[i].x;
    //             tmpPt.y = input_src_sem_cloud->points[i].y;
    //             tmpPt.z = input_src_sem_cloud->points[i].z;
    //             selected_src_points_->points.push_back(tmpPt);
    //         }//为什么要分src和tgt？
    //     }

    //     for (int i = 0; i < input_tgt_sem_cloud->points.size(); i++) {
    //         if (input_tgt_sem_cloud->points[i].label == params_.semanticLabel) {
    //             PointType tmpPt;
    //             tmpPt.x = input_tgt_sem_cloud->points[i].x;
    //             tmpPt.y = input_tgt_sem_cloud->points[i].y;
    //             tmpPt.z = input_tgt_sem_cloud->points[i].z;
    //             selected_tgt_points_->points.push_back(tmpPt);
    //         }
    //     }

    //     if (selected_src_points_->points.size() < 0 ||
    //         selected_tgt_points_->points.size() < 0)
    //         return false;

    //     return true;
    // }

    // following code for dynamic voxel segmentation
    //动态弯曲提速分割
    bool segmentPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input_sem_cloud) {
        // pointCloudL -> pointCloud
        //对于第一步为分割点云并保存在类的点云中
        if (!selectSemanticPoints(input_sem_cloud)) {
            ROS_ERROR("Select semantic points failed!");
            return false;
        } else {
            // step1, scan->polar coordinate
            //转换为极坐标
            convert2polar();
        }

        // step 2, create hash table
        //构建哈希表
        createHashTable();

        // step 3, DCVC segmentation
        //DCVC分割
        std::vector<int> labelInfo{};
        if (!DCVC(labelInfo)) {
            // ROS_ERROR("DCVC algorithm segmentation failure");
            return false;
        }

        // step 4, store segmentation results to clusters_
        //返回结果
        labelAnalysis(labelInfo);

        return true;
    }



    void convert2polar() {
        if (selected_points_->points.size() == 0) {
            // ROS_ERROR("Point cloud empty in converting cartesian to polar!");
        }

        // culculate yaw angle(rad)
        auto azimuthCal = [&](double x, double y) -> double {
            auto angle = static_cast<double>(std::atan2(y, x));
            return angle > 0.0 ? angle * 180 / M_PI
                               : (angle + 2 * M_PI) * 180 / M_PI;
        };

        size_t totalSize = selected_points_->points.size();
        polarCor.resize(totalSize);

        Eigen::Vector3d cur = Eigen::Vector3d::Zero();
        for (size_t i = 0; i < totalSize; ++i) {
            // polar pitch azimuth
            Eigen::Vector3d rpa = Eigen::Vector3d::Zero();
            cur(0) = selected_points_->points[i].x;
            cur(1) = selected_points_->points[i].y;
            cur(2) = selected_points_->points[i].z;
            rpa.x() = cur.norm();
            rpa.y() = std::asin(cur.z() / rpa.x()) * 180.0 / M_PI;
            rpa.z() = azimuthCal(cur.x(), cur.y());

            if (rpa.x() >= 120.0 || rpa.x() <= 0.5) continue;

            minPitch = rpa.y() < minPitch ? rpa.y() : minPitch;
            maxPitch = rpa.y() > maxPitch ? rpa.y() : maxPitch;
            minPolar = rpa.x() < minPolar ? rpa.x() : minPolar;
            maxPolar = rpa.x() > maxPolar ? rpa.x() : maxPolar;

            polarCor[i] = rpa;
        }

        polarCor.shrink_to_fit();

        polarNum = 0;
        polarBounds.clear();
        width = static_cast<int>(std::round(360.0 / params_.deltaA) + 1);
        height = static_cast<int>((maxPitch - minPitch) / params_.deltaP);
        double range = minPolar;
        int step = 1;
        while (range <= maxPolar) {
            range += (params_.startR - step * params_.deltaR);
            polarBounds.emplace_back(range);
            polarNum++, step++;
        }
    }

    //构建hash表
    void createHashTable() {
        size_t totalSize = polarCor.size();

        Eigen::Vector3d cur = Eigen::Vector3d::Zero();
        int polarIndex, pitchIndex, azimuthIndex, voxelIndex;
        voxelMap.reserve(totalSize);

        for (size_t item = 0; item < totalSize; ++item) {
            cur = polarCor[item];
            polarIndex = getPolarIndex(cur.x());
            pitchIndex = static_cast<int>(
                    std::round((cur.y() - minPitch) / params_.deltaP));
            azimuthIndex =
                    static_cast<int>(std::round(cur.z() / params_.deltaA));

            voxelIndex = (azimuthIndex * (polarNum + 1) + polarIndex) +
                         pitchIndex * (polarNum + 1) * (width + 1);

            auto iter = voxelMap.find(voxelIndex);
            if (iter != voxelMap.end()) {
                // iter->second.index.emplace_back(item);
                iter->second.emplace_back(item);
            } else {
                std::vector<int> index{};
                index.emplace_back(item);
                voxelMap.insert(std::make_pair(voxelIndex, index));
            }
        }
    }
    /**
    查询极坐标序号？索引？
     * @brief get the index value in the polar radial direction
     * @param radius, polar diameter
     * @return polar diameter index
     */
    int getPolarIndex(double& radius) {
        for (auto r = 0; r < polarNum; ++r) {
            if (radius < polarBounds[r]) return r;
        }
        return polarNum - 1;
    }
    /**
    重头戏，主要程序
     * @brief the Dynamic Curved-Voxle Clustering algoithm for fast and precise
     * point cloud segmentaiton
     * @param label_info, output the category information of each point
     * @return true if success otherwise false
     */
    bool DCVC(std::vector<int>& label_info) {
        int labelCount = 0;
        size_t totalSize = polarCor.size();
        if (totalSize <= 0) {
            // ROS_ERROR(
            //         "points in the cloud not enough to complete the DCVC "
            //         "algorithm");
            return false;
        }

        label_info.resize(totalSize, -1);
        Eigen::Vector3d cur = Eigen::Vector3d::Zero();
        int polar_index, pitch_index, azimuth_index, voxel_index, currInfo,
                neighInfo;

        for (size_t i = 0; i < totalSize; ++i) {
            if (label_info[i] != -1) continue;
            cur = polarCor[i];

            polar_index = getPolarIndex(cur.x());
            pitch_index = static_cast<int>(
                    std::round((cur.y() - minPitch) / params_.deltaP));
            azimuth_index =
                    static_cast<int>(std::round(cur.z() / params_.deltaA));
            voxel_index = (azimuth_index * (polarNum + 1) + polar_index) +
                          pitch_index * (polarNum + 1) * (width + 1);

            auto iter_find = voxelMap.find(voxel_index);
            std::vector<int> neighbors;
            if (iter_find != voxelMap.end()) {
                std::vector<int> KNN{};
                searchKNN(polar_index, pitch_index, azimuth_index, KNN);

                for (auto& k : KNN) {
                    iter_find = voxelMap.find(k);

                    if (iter_find != voxelMap.end()) {
                        neighbors.reserve(iter_find->second.size());
                        for (auto& id : iter_find->second) {
                            neighbors.emplace_back(id);
                        }
                    }
                }
            }

            neighbors.swap(neighbors);

            if (!neighbors.empty()) {
                for (auto& id : neighbors) {
                    currInfo = label_info[i];    // current label index
                    neighInfo = label_info[id];  // voxel label index
                    if (currInfo != -1 && neighInfo != -1 &&
                        currInfo != neighInfo) {
                        for (auto& seg : label_info) {
                            if (seg == currInfo) seg = neighInfo;
                        }
                    } else if (neighInfo != -1) {
                        label_info[i] = neighInfo;
                    } else if (currInfo != -1) {
                        label_info[id] = currInfo;
                    } else {
                        continue;
                    }
                }
            }

            // If there is no category information yet, then create a new label
            // information
            if (label_info[i] == -1) {
                labelCount++;
                label_info[i] = labelCount;
                for (auto& id : neighbors) {
                    label_info[id] = labelCount;
                }
            }
        }

        // free memory
        std::vector<Eigen::Vector3d,
                    Eigen::aligned_allocator<Eigen::Vector3d>>()
                .swap(polarCor);

        return true;
    }
    /**
    最近邻查找
     * @brief search for neighboring voxels
     * @param polar_index, polar diameter index
     * @param pitch_index, pitch angular index
     * @param azimuth_index, azimuth angular index
     * @param out_neighIndex, output adjacent voxel index set
     * @return void
     */
    void searchKNN(int& polar_index,
                   int& pitch_index,
                   int& azimuth_index,
                   std::vector<int>& out_neighIndex) const {
        for (auto z = pitch_index - 1; z <= pitch_index + 1; ++z) {
            if (z < 0 || z > height) continue;
            for (int y = polar_index - 1; y <= polar_index + 1; ++y) {
                if (y < 0 || y > polarNum) continue;

                for (int x = azimuth_index - 1; x <= azimuth_index + 1; ++x) {
                    int ax = x;
                    if (ax < 0) ax = width - 1;
                    if (ax > 300) ax = 300;

                    out_neighIndex.emplace_back((ax * (polarNum + 1) + y) +
                                                z * (polarNum + 1) *
                                                        (width + 1));
                }
            }
        }
    }

    /**
    删除包含较少点数的聚类，并将符合条件的聚类存储到一个点云向量中
     * @brief delete clusters with fewer points, store clusters into a vector of
     * point clouds
     * @param label_info, input category information
     * @return void
     */
    void labelAnalysis(std::vector<int>& label_info) {
        std::unordered_map<int, std::vector<int>> label2segIndex;
        size_t totalSize = label_info.size();
        for (size_t i = 0; i < totalSize; ++i) {
            // zero initialization for unordered_map
            label2segIndex[label_info[i]].emplace_back(i);
            // if (label2segIndex.find(label_info[i]) == label2segIndex.end()) {
            //     label2segIndex[label_info[i]].emplace_back(i);
            // }
            // else {
            //     label2segIndex[label_info[i]] += 1;
            // }
        }

        for (auto& it : label2segIndex) {
            if (it.second.size() >= params_.minSeg) {
                pcl::PointCloud<pcl::PointXYZ>::Ptr cur_cloud(
                        new pcl::PointCloud<pcl::PointXYZ>);
                for (auto& idx : it.second) {
                    // cur_cloud->points.emplace_back(selected_points_->points[idx]);
                    cur_cloud->points.push_back(selected_points_->points[idx]);
                }
                clusters_.push_back(cur_cloud);
            }
        }
        // free memory
        std::unordered_map<int, std::vector<int>>().swap(label2segIndex);
    }
    //参数设置
    void setParams(int semantic_class,
                   double cluster_distance_threshold,
                   int minNum,
                   int maxNum,
                   clusterManager::ClusterParams& params) {
        params.semanticLabel = semantic_class;
        params.clusterTolerance = cluster_distance_threshold;
        params.minClusterSize = minNum;
        params.maxClusterSize = maxNum;
    }

    void setParams(int semantic_class,
                   double cluster_distance_threshold,
                   int minNum,
                   int maxNum,
                   clusterManager::ClusterParams& params,
                   clusterManager::DCVCParam& seg_param) {
        params.semanticLabel = semantic_class;
        params.clusterTolerance = cluster_distance_threshold;
        params.minClusterSize = minNum;
        params.maxClusterSize = maxNum;

        params.startR = seg_param.startR;
        params.deltaR = seg_param.deltaR;
        params.deltaP = seg_param.deltaP;
        params.deltaA = seg_param.deltaA;
        params.minSeg = seg_param.minSeg;
    }
    // pcl::PointCloud<PointType>::Ptr getSrcMatchedPointCloud() {
    //     return src_matched_pcl_.makeShared();
    // }

    // pcl::PointCloud<PointType>::Ptr getTgtMatchedPointCloud() {
    //     return tgt_matched_pcl_.makeShared();
    // }

    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
    getSrcCovMat() {
        return covariances_src_;
    }

    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
    getTgtCovMat() {
        return covariances_tgt_;
    }
    
    // private:
    ClusterParams params_;

    // DBScan/Hierarchical DBScan related
    pcl::PointCloud<pcl::PointXYZ>::Ptr selected_points_;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_;
    std::vector<pcl::PointIndices> cluster_indices_;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec_;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr,
                Eigen::aligned_allocator<pcl::PointCloud<pcl::PointXYZ>::Ptr>>
            clusters_;

    // curved voxelization related
    double minPitch{0.0};
    double maxPitch{0.0};
    double minPolar{5.0};
    double maxPolar{5.0};
    int width{0};
    int height{0};
    // int minSeg{0};
    int polarNum{0};
    std::vector<double> polarBounds{};
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
            polarCor;
    std::unordered_map<int, std::vector<int>> voxelMap{};

    // fpfh related
    // pcl::PointCloud<PointType>::Ptr selected_src_points_;
    // pcl::PointCloud<PointType>::Ptr selected_tgt_points_;

    // teaser::PointCloud src_cloud_, tgt_cloud_;
    // teaser::FPFHCloudPtr src_descriptors_;
    // teaser::FPFHCloudPtr tgt_descriptors_;
    std::vector<std::pair<int, int>> corr_;  // Correspondence

    Eigen::Matrix<double, 3, Eigen::Dynamic>
            src_matched_;  // matched fpfh feature points
    Eigen::Matrix<double, 3, Eigen::Dynamic> tgt_matched_;

    // pcl::PointCloud<PointType> src_matched_pcl_;
    // pcl::PointCloud<PointType> tgt_matched_pcl_;

    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
            covariances_src_;
    std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
            covariances_tgt_;

    // cov mat computing relateds
    // RegularizationMethod reg_method_;
};

#endif