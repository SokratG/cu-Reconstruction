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
using PointTI = pcl::PointXYZI;
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

struct PCLSiftConfig
{
    r32 min_scale = 0.1f;
    i32 n_octaves = 6;
    i32 n_scales_per_octave = 10;
    r32 min_contrast = 0.5f;
};

struct PCLDescriptorConfig
{
    r32 normal_radius_search = 0.05;
    r32 feature_radius_search = 0.1;
    i32 inlier_size = 200;
    r32 inlier_threshold = 0.2;
};


PointCloudCPtr voxel_filter_pc(const PointCloudCPtr pcl_pc,
                               const VoxelFilterConfig& vfc = VoxelFilterConfig());


PointCloudCPtr stitch_icp_point_clouds(const std::vector<PointCloudCPtr>& pcl_pc,
                                      const ICPCriteria& icp_criteria = ICPCriteria());

PointCloudCPtr stitch_feature_registration_point_cloud(const std::vector<PointCloudCPtr>& pcl_pc,
                                                       const PCLSiftConfig& pcl_sift_cfg = PCLSiftConfig(),
                                                       const PCLDescriptorConfig& pcl_desc_cfg = PCLDescriptorConfig());

PointCloudCPtr build_point_cloud(const KeyFrame::Ptr rgb, 
                                 const KeyFrame::Ptr depth, 
                                 const Mat3& camera_matrix,
                                 const StatisticalFilterConfig& sfc = StatisticalFilterConfig());

PointCloudCPtr transform_point_cloud(const PointCloudCPtr pcl_pc, const SE3& T);


cudaPointCloud::Ptr pcl_to_cuda_pc(const PointCloudCPtr pcl_pc,
                                   const std::array<r64, 9>& K);




};

#endif