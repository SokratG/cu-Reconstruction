#ifndef CUREC_LIB_POINT_CLOUD_HPP
#define CUREC_LIB_POINT_CLOUD_HPP

#include "types.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>

namespace curec {


// PCL
using PointT = pcl::PointXYZ;
using PointTC = pcl::PointXYZRGB;
using PointCloud = pcl::PointCloud<PointT>;
using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;
using PointCloudC = pcl::PointCloud<PointTC>;
using PointCloudCPtr = pcl::PointCloud<PointTC>::Ptr;

class RawPointCloud {
public:
    RawPointCloud();

    PointCloudCPtr cloud;

    void add_point(const Vec3& pt, const Vec3f& color);
    void write_to_ply(const std::string& filepath);
private:    
};


};


#endif