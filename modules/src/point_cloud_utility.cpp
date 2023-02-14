#include "point_cloud_utility.hpp"

#include <pcl/filters/statistical_outlier_removal.h>

// #include <pcl/filters/voxel_grid.h>

#include <glog/logging.h>

namespace cuphoto {

PointCloudCPtr statistical_filter_pc(const PointCloudCPtr current_pc, const StatisticalFilterConfig& sfc) {
    pcl::StatisticalOutlierRemoval<PointTC> statistical_filter;
    statistical_filter.setMeanK(sfc.Kmean);
    statistical_filter.setStddevMulThresh(sfc.StddevMulThresh);
    statistical_filter.setInputCloud(current_pc);

    PointCloudCPtr filtered_pc(new PointCloudC);

    statistical_filter.filter(*filtered_pc);

    return filtered_pc;
}

void voxel_filter_pc(cudaPointCloud::Ptr& cuda_pc) {
    LOG(ERROR) << "IMPLEMENT HERE!";
    return;
}


PointCloudCPtr build_point_cloud(const KeyFrame::Ptr rgb, 
                                 const KeyFrame::Ptr depth, 
                                 const Mat3& camera_matrix,
                                 const StatisticalFilterConfig& sfc) {
    const auto width = depth->frame().cols;
    const auto height = depth->frame().rows;
    cv::Mat depth_frame;
    depth->frame().download(depth_frame);

    const r64 fx = camera_matrix(0, 0);
    const r64 fy = camera_matrix(1, 1);
    const r64 cx = camera_matrix(0, 2);
    const r64 cy = camera_matrix(1, 2);
    PointCloudCPtr current_pc(new PointCloudC);
    SE3 T = rgb->pose();
    for (auto v = 0; v < height; ++v) {
        for (auto u = 0; u < width; ++u) {
            const r32 d = depth_frame.ptr<r32>(v)[u];
            if (d <= sfc.depth_threshold)
                continue;
            Vec3 point;
            point[2] = static_cast<r64>(d);
            point[0] = (u - cx) * point[2] / fx;
            point[1] = (v - cy) * point[2] / fy;
            Vec3 pointWorld = T * point;
            PointTC p;
            p.x = pointWorld[0];
            p.y = pointWorld[1];
            p.z = pointWorld[2];
            Vec3f color = rgb->get_color(cv::Point2f(u, v));
            p.r = color.x();
            p.g = color.y();
            p.b = color.z();
            current_pc->points.push_back(p);
        }
    }

    const auto filtered_pc = statistical_filter_pc(current_pc, sfc);

    return filtered_pc;
}


cudaPointCloud::Ptr pcl_to_cuda_pc(const PointCloudCPtr pcl_pc,
                                   const std::array<r64, 9>& K) {
    cudaPointCloud::Ptr cpc = cudaPointCloud::create(K, pcl_pc->size());
    for (auto idx = 0; idx < cpc->get_total_num_points(); ++idx) {
        // for opengl / meshlab mult by -1.f
        float3 pos = make_float3(pcl_pc->points[idx].x, pcl_pc->points[idx].y * -1.f, pcl_pc->points[idx].z * -1.f);
        uchar3 color = make_uchar3(pcl_pc->points[idx].r, pcl_pc->points[idx].g, pcl_pc->points[idx].b);
        cpc->add_vertex(pos, color, idx);
    }

    cpc->set_total_number_pts();
    return cpc;
}


void stitch_icp_point_cloud(const PointCloudCPtr pcl_pc_query, const PointCloudCPtr pcl_pc_target) {
    // TODO
    LOG(ERROR) << "IMPLEMENT HERE!";
    return;
}

void stitch_feature_registration_point_cloud(const PointCloudCPtr pcl_pc_query, const PointCloudCPtr pcl_pc_target) {
    LOG(ERROR) << "IMPLEMENT HERE!";
    return;
}


};