#include "point_cloud_utility.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

// #include <pcl/filters/voxel_grid.h>
// #include <pcl/filters/statistical_outlier_removal.h>


#include <glog/logging.h>

namespace cuphoto {

// PCL
using PointT = pcl::PointXYZ;
using PointTC = pcl::PointXYZRGB;
using PointCloud = pcl::PointCloud<PointT>;
using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;
using PointCloudC = pcl::PointCloud<PointTC>;
using PointCloudCPtr = pcl::PointCloud<PointTC>::Ptr;

void statistical_filter_pc(cudaPointCloud::Ptr& cuda_pc) {
    LOG(ERROR) << "IMPLEMENT HERE!";
    return;
}

void voxel_filter_pc(cudaPointCloud::Ptr& cuda_pc) {
    LOG(ERROR) << "IMPLEMENT HERE!";
    return;
}


cudaPointCloud::Ptr pcl_to_cuda_pc(const KeyFrame::Ptr rgb, const KeyFrame::Ptr depth, const Camera::Ptr camera) {
    std::array<r64, 9> K;
    Vec9::Map(K.data()) = Eigen::Map<Vec9>(camera->K().data(), camera->K().cols() * camera->K().rows());
    
    return cudaPointCloud::create(K, 0);
}

};