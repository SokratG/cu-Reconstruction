#include "point_cloud_utility.hpp"

#include <pcl/common/transforms.h>

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>

#include <memory>
#include <cmath>

#include <glog/logging.h>

namespace cuphoto {


PointCloudCPtr statistical_filter_pc(const PointCloudCPtr current_pc, const StatisticalFilterConfig& sfc) {
    pcl::StatisticalOutlierRemoval<PointTC> statistical_filter;
    statistical_filter.setMeanK(sfc.k_mean);
    statistical_filter.setStddevMulThresh(sfc.std_dev_mul_thresh);
    statistical_filter.setInputCloud(current_pc);

    PointCloudCPtr filtered_pc(new PointCloudC);

    statistical_filter.filter(*filtered_pc);

    return filtered_pc;
}

PointCloudCNPtr statistical_filter_pc(const PointCloudCNPtr current_pc, const StatisticalFilterConfig& sfc) {
    pcl::StatisticalOutlierRemoval<PointTCN> statistical_filter;
    statistical_filter.setMeanK(sfc.k_mean);
    statistical_filter.setStddevMulThresh(sfc.std_dev_mul_thresh);
    statistical_filter.setInputCloud(current_pc);

    PointCloudCNPtr filtered_pc(new PointCloudCN);

    statistical_filter.filter(*filtered_pc);

    return filtered_pc;
}


PointCloudPtr voxel_filter_pc(const PointCloudPtr pcl_pc,
                              const VoxelFilterConfig& vfc) {
    pcl::VoxelGrid<PointT> voxel_filter;
    voxel_filter.setLeafSize(vfc.resolution, vfc.resolution, vfc.resolution);
    voxel_filter.setInputCloud(pcl_pc);
    PointCloudPtr filtered_pc(new PointCloud);
    voxel_filter.filter(*filtered_pc);
    return filtered_pc;
}

PointCloudCPtr voxel_filter_pc(const PointCloudCPtr pcl_pc,
                               const VoxelFilterConfig& vfc) {
    pcl::VoxelGrid<PointTC> voxel_filter;
    voxel_filter.setLeafSize(vfc.resolution, vfc.resolution, vfc.resolution);
    voxel_filter.setInputCloud(pcl_pc);
    PointCloudCPtr filtered_pc(new PointCloudC);
    voxel_filter.filter(*filtered_pc);
    return filtered_pc;
}

PointCloudCNPtr voxel_filter_pc(const PointCloudCNPtr pcl_pc,
                               const VoxelFilterConfig& vfc) {
    pcl::VoxelGrid<PointTCN> voxel_filter;
    voxel_filter.setLeafSize(vfc.resolution, vfc.resolution, vfc.resolution);
    voxel_filter.setInputCloud(pcl_pc);
    PointCloudCNPtr filtered_pc(new PointCloudCN);
    voxel_filter.filter(*filtered_pc);
    return filtered_pc;
}


PointCloudCPtr point_cloud_from_depth(const KeyFrame::Ptr rgb, 
                                      const KeyFrame::Ptr depth, 
                                      const Mat3& camera_matrix,
                                      const r32 depth_threshold_min,
                                      const r32 depth_threshold_max) {
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
            if (d <= depth_threshold_min || d >= depth_threshold_max)
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

    return current_pc;
}


cudaPointCloud::Ptr pcl_to_cuda_pc(const PointCloudCPtr pcl_pc,
                                   const std::array<r64, 9>& K) {
    cudaPointCloud::Ptr cpc = cudaPointCloud::create(K, pcl_pc->size());
    for (auto idx = 0; idx < cpc->get_total_num_points(); ++idx) {
        float3 pos = make_float3(pcl_pc->points[idx].x, pcl_pc->points[idx].y, pcl_pc->points[idx].z);
        uchar3 color = make_uchar3(pcl_pc->points[idx].r, pcl_pc->points[idx].g, pcl_pc->points[idx].b);
        cpc->add_vertex(pos, color, idx);
    }

    cpc->set_total_number_pts();
    return cpc;
}

cudaPointCloud::Ptr pcl_to_cuda_pc(const PointCloudPtr pcl_pc,
                                   const std::array<r64, 9>& K) {
    cudaPointCloud::Ptr cpc = cudaPointCloud::create(K, pcl_pc->size());
    for (auto idx = 0; idx < cpc->get_total_num_points(); ++idx) {
        float3 pos = make_float3(pcl_pc->points[idx].x, pcl_pc->points[idx].y, pcl_pc->points[idx].z);
        uchar3 color = make_uchar3(255, 255, 255);
        cpc->add_vertex(pos, color, idx);
    }

    cpc->set_total_number_pts();
    return cpc;
}


cudaPointCloud::Ptr pcl_to_cuda_pc(const PointCloudCPtr pcl_pc,
                                   const PointCloudNPtr pcl_normals,
                                   const std::array<r64, 9>& K) {
    cudaPointCloud::Ptr cpc = cudaPointCloud::create(K, pcl_pc->size());
    for (auto idx = 0; idx < cpc->get_total_num_points(); ++idx) {
        float3 pos = make_float3(pcl_pc->points[idx].x, pcl_pc->points[idx].y, pcl_pc->points[idx].z);
        uchar3 color = make_uchar3(pcl_pc->points[idx].r, pcl_pc->points[idx].g, pcl_pc->points[idx].b);
        float4 normal = make_float4(pcl_normals->points[idx].normal_x, pcl_normals->points[idx].normal_y,
                                    pcl_normals->points[idx].normal_z, pcl_normals->points[idx].curvature);
        cpc->add_vertex(pos, color, idx, normal);
    }

    cpc->set_total_number_pts();
    return cpc;
}


PointCloudCPtr cuda_pc_to_pcl(const cudaPointCloud::Ptr cuda_pc) {
    PointCloudCPtr pcl_pc(new PointCloudC);
    for (auto idx = 0; idx < cuda_pc->get_total_num_points(); ++idx) {
        cudaPointCloud::Vertex* v = cuda_pc->get_vertex(idx);
        PointTC p;
        p.x = v->pos.x;
        p.y = v->pos.y;
        p.z = v->pos.z;
        p.r = v->color.x;
        p.g = v->color.y;
        p.b = v->color.z;
        pcl_pc->points.push_back(p);
    }
    return pcl_pc;
}


PointCloudCPtr transform_point_cloud(const PointCloudCPtr pcl_pc, const Mat4& T) {
    PointCloudCPtr pcl_target(new PointCloudC);
    pcl::transformPointCloud(*pcl_pc, *pcl_target, T);
    return pcl_target;
}


PointCloudCPtr transform_point_cloud(const PointCloudCPtr pcl_pc, const SE3& T) {
    PointCloudCPtr pcl_target(new PointCloudC);
    pcl::transformPointCloud(*pcl_pc, *pcl_target, T.matrix());
    return pcl_target;
}

PointCloudCNPtr transform_point_cloud(const PointCloudCNPtr pcl_pc, const Mat4& T) {
    PointCloudCNPtr pcl_target(new PointCloudCN);
    pcl::transformPointCloud(*pcl_pc, *pcl_target, T);
    return pcl_target;
}

PointCloudCNPtr transform_point_cloud(const PointCloudCNPtr pcl_pc, const SE3& T) {
    PointCloudCNPtr pcl_target(new PointCloudCN);
    pcl::transformPointCloud(*pcl_pc, *pcl_target, T.matrix());
    return pcl_target;
}


cudaPointCloud::Ptr compute_normals_pc(const cudaPointCloud::Ptr cu_pc, const r64 radius_search, const i32 k_nn) {
    const auto pcl_pc = cuda_pc_to_pcl(cu_pc);
    pcl::NormalEstimationOMP<PointTC, pcl::Normal> ne;
    pcl::search::KdTree<PointTC>::Ptr tree(new pcl::search::KdTree<PointTC>());
    PointCloudNPtr cloud_normals(new PointCloudN);

    ne.setSearchMethod(tree);
    ne.setRadiusSearch(radius_search);
    ne.setKSearch(k_nn);
    ne.setInputCloud(pcl_pc);
    ne.compute(*cloud_normals);

    PointCloudCPtr filter_pc(new PointCloudC);
    PointCloudNPtr filter_normals(new PointCloudN);
    pcl::PointIndices::Ptr pt_inds(new pcl::PointIndices());

    pcl::removeNaNNormalsFromPointCloud(*cloud_normals, *filter_normals, pt_inds->indices);
    pcl::ExtractIndices<PointTC> extract;
    extract.setInputCloud(pcl_pc);
    extract.setIndices(pt_inds);
    extract.filter(*filter_pc);

    cudaPointCloud::Ptr cuda_normal_pc = pcl_to_cuda_pc(filter_pc, filter_normals, cu_pc->get_camera_parameters());

    return cuda_normal_pc;
}



void transform_cuda_pc(cudaPointCloud::Ptr& cu_pc, const SE3& pose) {
    Mat3 R = pose.rotationMatrix();
    Vec3 t = pose.translation();
    cuphoto::Quat q(R);
    std::array<r64, 7> T {q.w(), q.x(), q.y(), q.z(), t.x(), t.y(), t.z()};
    cu_pc->transform(T);
}


};