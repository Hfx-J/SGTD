//
// Created by xiang on 2022/7/14.
//

#ifndef SLAM_IN_AUTO_DRIVING_NDT_3D_H
#define SLAM_IN_AUTO_DRIVING_NDT_3D_H

#include "eigen_types.h"
// #include "common/point_types.h"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <unordered_map>

namespace sad {

// 定义系统中用到的点和点云类型
 using PointType = pcl::PointXYZ;         //wx-
//using PointType = pcl::PointXYZINormal;

using PointCloudType = pcl::PointCloud<PointType>;
using CloudPtr = PointCloudType::Ptr;


// 点云到Eigen的常用的转换函数
inline Vec3f ToVec3f(const PointType& pt) { return pt.getVector3fMap(); }
inline Vec3d ToVec3d(const PointType& pt) { return pt.getVector3fMap().cast<double>(); }

// 生成二维旋转矩阵
inline Eigen::Matrix4f getRotationMatrix(double angle) {
    double radians = angle * M_PI / 180.0;
    Eigen::Matrix4f rotationMatrix = Eigen::Matrix4f::Identity();
    rotationMatrix(0, 0) = cos(radians);
    rotationMatrix(0, 1) = -sin(radians);
    rotationMatrix(1, 0) = sin(radians);
    rotationMatrix(1, 1) = cos(radians);
    return rotationMatrix;
}

// 生成新的变换矩阵
inline Eigen::Matrix4f getNewTransformationMatrix(const Eigen::Matrix4f &originalMatrix, double angle) {
    Eigen::Matrix4f rotationMatrix = getRotationMatrix(angle);
    Eigen::Matrix4f newMatrix = rotationMatrix * originalMatrix;
    return newMatrix;
}
















/**
 * 计算一个容器内数据的均值与矩阵形式协方差
 * @tparam C    容器类型
 * @tparam int 　数据维度
 * @tparam Getter   获取数据函数, 接收一个容器内数据类型，返回一个Eigen::Matrix<double, dim,1> 矢量类型
 */
template <typename C, int dim, typename Getter>
inline void ComputeMeanAndCov(const C &data, Eigen::Matrix<double, dim, 1> &mean, Eigen::Matrix<double, dim, dim> &cov,
                       Getter &&getter)
{
    using D = Eigen::Matrix<double, dim, 1>;
    using E = Eigen::Matrix<double, dim, dim>;
    size_t len = data.size();

    assert(len > 1); //////////

    // clang-format off
    mean = std::accumulate(data.begin(), data.end(), Eigen::Matrix<double, dim, 1>::Zero().eval(),
                           [&getter](const D& sum, const auto& data) -> D { return sum + getter(data); }) / len;
    cov = std::accumulate(data.begin(), data.end(), E::Zero().eval(),
                          [&mean, &getter](const E& sum, const auto& data) -> E {
                                                                                                                                                                                                                           D v = getter(data) - mean;
                              return sum + v * v.transpose();
                          }) / (len - 1);
    // clang-format on
}
inline bool compareBySecond(const std::pair<double, double> &a, const std::pair<double, double> &b) {
    return a.second < b.second;
}
inline bool compareBySecond3(const std::vector<double> &a, const std::vector<double> &b) {
    // if (a[2]!=b[2] )return a[2] < b[2];
    // else 
    return a[1] > b[1];
    // return a[1]*(a[2]*10+1) > b[1]*(b[2]*10+1);
}
inline bool compareBySecond4(const std::vector<double> &a, const std::vector<double> &b) {
    return a[1] > b[1];
    // else 
    // return a[1] < b[1];
}

/**
 * 3D 形式的NDT
 */
class Ndt3d {
   public:
   
    enum class NearbyType {
        CENTER,   // 只考虑中心
        NEARBY6,  // 上下左右前后
    };

    struct Options {
        int max_iteration_ = 40;        // 最大迭代次数
        double voxel_size_ = 1.0;       // 体素大小
        double inv_voxel_size_ = 1.0;   //
        int min_effective_pts_ = 5;    // 最近邻点数阈值
        int min_pts_in_voxel_ = 3;      // 每个栅格中最小点数
        double eps_ = 1e-2;             // 收敛判定条件
        double res_outlier_th_ = 20.0;  // 异常值拒绝阈值
        bool remove_centroid_ = false;  // 是否计算两个点云中心并移除中心？

        NearbyType nearby_type_ = NearbyType::NEARBY6;
    };

    using KeyType = Eigen::Matrix<int, 3, 1>;  // 体素的索引
    struct VoxelData {
        VoxelData() {}
        VoxelData(size_t id) { idx_.emplace_back(id); }

        std::vector<size_t> idx_;      // 点云中点的索引
        Vec3d mu_ = Vec3d::Zero();     // 均值
        Mat3d sigma_ = Mat3d::Zero();  // 协方差
        Mat3d info_ = Mat3d::Zero();   // 协方差之逆
    };

    Ndt3d() {
        options_.inv_voxel_size_ = 1.0 / options_.voxel_size_;
        GenerateNearbyGrids();
    }

    Ndt3d(Options options) : options_(options) {
        options_.inv_voxel_size_ = 1.0 / options_.voxel_size_;
        GenerateNearbyGrids();
    }

    /// 设置目标的Scan
    void SetTarget(CloudPtr target) {
        target_ = target;
        BuildVoxels();
        // std::cout <<"222222222222"<<std::endl;

        // 计算点云中心
        target_center_ = std::accumulate(target->points.begin(), target_->points.end(), Vec3d::Zero().eval(),
                                         [](const Vec3d& c, const PointType& pt) -> Vec3d { return c + ToVec3d(pt); }) /
                         target_->size();
    }

    /// 设置被配准的Scan
    void SetSource(CloudPtr source) {
        source_ = source;

        source_center_ = std::accumulate(source_->points.begin(), source_->points.end(), Vec3d::Zero().eval(),
                                         [](const Vec3d& c, const PointType& pt) -> Vec3d { return c + ToVec3d(pt); }) /
                         source_->size();
    }

    void SetGtPose(const SE3& gt_pose) {
        gt_pose_ = gt_pose;
        gt_set_ = true;
    }

    /// 使用gauss-newton方法进行ndt配准
    bool AlignNdt(SE3& init_pose);
    Eigen::Matrix3d Feng_NDT_LOCATLIZATION(double initial_x,double initial_y,double initial_z);
    double ndt_iter_ = 0;
    double ndt_score =0;            //新增匹配得分-0706
    double  ndt_dx_= 0.0;
    bool ndt_ailgn_ = false;
    double ndt_matching_score_ = 0.0;
    

   private:
    void BuildVoxels();

    /// 根据最近邻的类型，生成附近网格
    void GenerateNearbyGrids();

    CloudPtr target_ = nullptr;
    CloudPtr source_ = nullptr;

    Vec3d target_center_ = Vec3d::Zero();
    Vec3d source_center_ = Vec3d::Zero();

    SE3 gt_pose_;
    bool gt_set_ = false;

    Options options_;

    std::unordered_map<KeyType, VoxelData, hash_vec<3>> grids_;  // 栅格数据
    std::vector<KeyType> nearby_grids_;                          // 附近的栅格
};

}  // namespace sad


#endif  // SLAM_IN_AUTO_DRIVING_NDT_3D_H
