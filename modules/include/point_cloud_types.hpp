#ifndef CUPHOTO_LIB_POINT_CLOUD_TYPES_HPP
#define CUPHOTO_LIB_POINT_CLOUD_TYPES_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace cuphoto {


// PCL
using PointT = pcl::PointXYZ;
using PointTI = pcl::PointXYZI;
using PointTC = pcl::PointXYZRGB;
using PointTCN = pcl::PointXYZRGBNormal;
using PointCloud = pcl::PointCloud<PointT>;
using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;
using PointCloudC = pcl::PointCloud<PointTC>;
using PointCloudCPtr = pcl::PointCloud<PointTC>::Ptr;
using PointCloudCN = pcl::PointCloud<PointTCN>;
using PointCloudCNPtr = pcl::PointCloud<PointTCN>::Ptr;


};

#endif