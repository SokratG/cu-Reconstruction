#include "point_cloud.hpp"
#include <pcl/io/ply_io.h>

namespace curec {

RawPointCloud::RawPointCloud() : cloud(new PointCloudC) {

}

void RawPointCloud::add_point(const Vec3& pt, const Vec3f& color) {
    PointTC ptc;
    ptc.x = pt.x();
    ptc.y = pt.y();
    ptc.z = pt.z();
    ptc.r = color.x();
    ptc.g = color.x();
    ptc.b = color.z();
    cloud->points.push_back(ptc);
}

void RawPointCloud::write_to_ply(const std::string& filepath) {
    pcl::PLYWriter writer;
    // TODO: use_binary=true
    writer.write(filepath, *cloud, false, false);
}

};