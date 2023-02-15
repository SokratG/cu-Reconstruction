#ifndef CUPHOTO_LIB_POINT_CLOUD_FILTER_HPP
#define CUPHOTO_LIB_POINT_CLOUD_FILTER_HPP

#include "cuda/cuda_point_cloud.cuh"
#include "keyframe.hpp"
#include "camera.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <array>

namespace cuphoto {


// PCL
using PointT = pcl::PointXYZ;
using PointTC = pcl::PointXYZRGB;
using PointCloud = pcl::PointCloud<PointT>;
using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;
using PointCloudC = pcl::PointCloud<PointTC>;
using PointCloudCPtr = pcl::PointCloud<PointTC>::Ptr;

struct StatisticalFilterConfig
{
    i32 k_mean = 15;
    r32 std_dev_mul_thresh = 1.1;
    r32 depth_threshold = 1e-8;
};

struct VoxelFilterConfig 
{
    r64 resolution = 0.05;
    ui32 min_points_per_voxel = -1; 
};

struct ICPCriteria
{
    r64 max_correspond_dist = 0.05;
    r64 transformation_eps = 1e-8;
    i32 max_iteration = 10;
};


PointCloudCPtr voxel_filter_pc(const PointCloudCPtr pcl_pc,
                               const VoxelFilterConfig& vfc = VoxelFilterConfig());


PointCloudCPtr stitch_icp_point_clouds(const std::vector<PointCloudCPtr>& pcl_pc,
                                      const ICPCriteria& icp_criteria = ICPCriteria());

void stitch_feature_registration_point_cloud(const PointCloudCPtr pcl_pc_query, const PointCloudCPtr pcl_pc_target);

PointCloudCPtr build_point_cloud(const KeyFrame::Ptr rgb, 
                                 const KeyFrame::Ptr depth, 
                                 const Mat3& camera_matrix,
                                 const StatisticalFilterConfig& sfc = StatisticalFilterConfig());

PointCloudCPtr transform_point_cloud(const PointCloudCPtr pcl_pc, const SE3& T);


cudaPointCloud::Ptr pcl_to_cuda_pc(const PointCloudCPtr pcl_pc,
                                   const std::array<r64, 9>& K);




};

#endif