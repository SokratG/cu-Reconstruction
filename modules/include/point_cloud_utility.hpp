#ifndef CUPHOTO_LIB_POINT_CLOUD_FILTER_HPP
#define CUPHOTO_LIB_POINT_CLOUD_FILTER_HPP

#include "cuda/cuda_point_cloud.cuh"
#include "keyframe.hpp"
#include "camera.hpp"

namespace cuphoto {

void statistical_filter_pc(cudaPointCloud::Ptr& cuda_pc);


void voxel_filter_pc(cudaPointCloud::Ptr& cuda_pc);

cudaPointCloud::Ptr pcl_to_cuda_pc(const KeyFrame::Ptr rgb, const KeyFrame::Ptr depth, const Camera::Ptr camera);

};

#endif