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
    i32 Kmean = 50;
    r32 StddevMulThresh = 1.0;
    r32 depth_threshold = 1e-8;
};


void voxel_filter_pc(cudaPointCloud::Ptr& cuda_pc);


void stitch_icp_point_cloud(const PointCloudCPtr pcl_pc_query, const PointCloudCPtr pcl_pc_target);

void stitch_feature_registration_point_cloud(const PointCloudCPtr pcl_pc_query, const PointCloudCPtr pcl_pc_target);

PointCloudCPtr build_point_cloud(const KeyFrame::Ptr rgb, 
                                 const KeyFrame::Ptr depth, 
                                 const Mat3& camera_matrix,
                                 const StatisticalFilterConfig& sfc = StatisticalFilterConfig());


cudaPointCloud::Ptr pcl_to_cuda_pc(const PointCloudCPtr pcl_pc,
                                   const std::array<r64, 9>& K);




};

#endif