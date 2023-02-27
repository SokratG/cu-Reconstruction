#ifndef CUPHOTO_LIB_POINT_CLOUD_UTILITY_HPP
#define CUPHOTO_LIB_POINT_CLOUD_UTILITY_HPP

#include "cuda/cuda_point_cloud.cuh"
#include "keyframe.hpp"
#include "camera.hpp"
#include "point_cloud_types.hpp"

#include <array>
#include <limits>

namespace cuphoto {


struct StatisticalFilterConfig
{
    i32 k_mean = 25;
    r32 std_dev_mul_thresh = 0.7;
};

struct VoxelFilterConfig 
{
    r64 resolution = 0.07;
    ui32 min_points_per_voxel = -1; 
};


PointCloudCPtr voxel_filter_pc(const PointCloudCPtr pcl_pc,
                               const VoxelFilterConfig& vfc = VoxelFilterConfig());
PointCloudCNPtr voxel_filter_pc(const PointCloudCNPtr pcl_pc,
                               const VoxelFilterConfig& vfc = VoxelFilterConfig());

PointCloudCPtr statistical_filter_pc(const PointCloudCPtr current_pc, 
                                     const StatisticalFilterConfig& sfc = StatisticalFilterConfig());

PointCloudCNPtr statistical_filter_pc(const PointCloudCNPtr current_pc, 
                                      const StatisticalFilterConfig& sfc = StatisticalFilterConfig());

PointCloudCPtr point_cloud_from_depth(const KeyFrame::Ptr rgb, 
                                      const KeyFrame::Ptr depth, 
                                      const Mat3& camera_matrix,
                                      const r32 depth_threshold_min = std::numeric_limits<r32>::min(),
                                      const r32 depth_threshold_max = std::numeric_limits<r32>::max());

PointCloudCPtr transform_point_cloud(const PointCloudCPtr pcl_pc, const SE3& T);
PointCloudCPtr transform_point_cloud(const PointCloudCPtr pcl_pc, const Mat4& T);
PointCloudCNPtr transform_point_cloud(const PointCloudCNPtr pcl_pc, const Mat4& T);
PointCloudCNPtr transform_point_cloud(const PointCloudCNPtr pcl_pc, const SE3& T);


cudaPointCloud::Ptr pcl_to_cuda_pc(const PointCloudCPtr pcl_pc,
                                   const std::array<r64, 9>& K);

PointCloudCPtr cuda_pc_to_pcl(const cudaPointCloud::Ptr pcl_pc);


};

#endif