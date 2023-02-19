#include "point_cloud_utility.hpp"

#include <pcl/common/transforms.h>

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/keypoints/sift_keypoint.h>

#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_estimation_lm.h>

#include <memory>
#include <cmath>

#include <glog/logging.h>

namespace cuphoto {


struct PCLMatchAdjacent {
    i32 src_idx;
    i32 dst_idx;
    std::vector<i32> correspondences;
    PCLMatchAdjacent() = delete;
    PCLMatchAdjacent(const i32 _src_idx) : src_idx(_src_idx), dst_idx(-1) {}
};



PointCloudCPtr statistical_filter_pc(const PointCloudCPtr current_pc, const StatisticalFilterConfig& sfc) {
    pcl::StatisticalOutlierRemoval<PointTC> statistical_filter;
    statistical_filter.setMeanK(sfc.k_mean);
    statistical_filter.setStddevMulThresh(sfc.std_dev_mul_thresh);
    statistical_filter.setInputCloud(current_pc);

    PointCloudCPtr filtered_pc(new PointCloudC);

    statistical_filter.filter(*filtered_pc);

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
            if (d <= sfc.depth_threshold_min || d >= sfc.depth_threshold_max)
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


PointCloudCPtr stitch_icp_point_clouds(const std::vector<PointCloudCPtr>& pcl_pc,
                                       const ICPCriteria& icp_criteria) {
    const auto pcl_query = pcl_pc.front();
    PointCloudCPtr filtered_pcl_pc_query = voxel_filter_pc(pcl_query);
    PointCloudCPtr total_pc(new PointCloudC);
    for (auto idx = 1; idx < pcl_pc.size(); ++idx) {
        pcl::IterativeClosestPoint<PointTC, PointTC> icp;
        const auto pcl_target = pcl_pc[idx];
        const PointCloudCPtr filtered_pcl_pc_target = voxel_filter_pc(pcl_target);
        icp.setInputSource(filtered_pcl_pc_query);
        icp.setInputTarget(filtered_pcl_pc_target);
        PointCloudC align_data;

        icp.setMaxCorrespondenceDistance(icp_criteria.max_correspond_dist);
        icp.setTransformationEpsilon(icp_criteria.transformation_eps);
        icp.setMaximumIterations(icp_criteria.max_iteration);

        icp.align(align_data);

        if (!icp.hasConverged()) {
            LOG(WARNING) << "stitching ICP has not converged!";
        }

        (*filtered_pcl_pc_query) = align_data;
        (*total_pc) += align_data;
    }
    
    
    return total_pc;
}

PointCloudCPtr stitch_feature_registration_point_cloud(const std::vector<PointCloudCPtr>& pcl_pc,
                                                       const PCLSiftConfig& pcl_sift_cfg,
                                                       const PCLDescriptorConfig& pcl_desc_cfg) {
    // voxel fitler
    std::vector<PointCloudCPtr> filtered_pcl_pc(pcl_pc.size());
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> filtered_normals(pcl_pc.size());
    for (auto idx = 0; idx < pcl_pc.size(); ++idx) {
        filtered_pcl_pc[idx] = voxel_filter_pc(pcl_pc.at(idx));
        pcl::NormalEstimationOMP<PointTC, pcl::Normal> normal_estimation;
        normal_estimation.setSearchMethod(pcl::search::Search<PointTC>::Ptr(new pcl::search::KdTree<PointTC>));
        normal_estimation.setRadiusSearch(pcl_desc_cfg.normal_radius_search);
        normal_estimation.setInputCloud(filtered_pcl_pc.at(idx));

        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        normal_estimation.compute(*normals);
        PointCloudCPtr filter_pc(new PointCloudC);
        pcl::PointCloud<pcl::Normal>::Ptr filter_normals(new pcl::PointCloud<pcl::Normal>);
        pcl::PointIndices::Ptr pt_inds(new pcl::PointIndices());
        pcl::removeNaNNormalsFromPointCloud(*normals, *filter_normals, pt_inds->indices);
        pcl::ExtractIndices<PointTC> extract;
        extract.setInputCloud(filtered_pcl_pc.at(idx));
        extract.setIndices(pt_inds);
        extract.filter(*filter_pc);

        filtered_pcl_pc[idx] = filter_pc;
        filtered_normals[idx] = filter_normals;
    }


    // keypoint detection
    pcl::SIFTKeypoint<PointTC, PointTC> sift;
    sift.setScales(pcl_sift_cfg.min_scale, pcl_sift_cfg.n_octaves, pcl_sift_cfg.n_scales_per_octave);
    sift.setMinimumContrast(pcl_sift_cfg.min_contrast);
    std::vector<PointCloudCPtr> keypoints;
    for (auto idx = 0; idx < filtered_pcl_pc.size(); ++idx) {
        PointCloudCPtr result(new PointCloudC);
        pcl::search::KdTree<PointTC>::Ptr tree(new pcl::search::KdTree<PointTC>);
        sift.setSearchMethod(tree);
        sift.setInputCloud(filtered_pcl_pc.at(idx));
        sift.compute(*result);
        keypoints.emplace_back(result);
    }

    // descriptor extraction
    pcl::FPFHEstimationOMP<PointTC, pcl::Normal, pcl::FPFHSignature33> feature_extractor;
    feature_extractor.setRadiusSearch(pcl_desc_cfg.feature_radius_search);
    std::vector<pcl::PointCloud<pcl::FPFHSignature33>::Ptr> features; 
    for (auto idx = 0; idx < keypoints.size(); ++idx) {
        feature_extractor.setSearchMethod(pcl::search::Search<PointTC>::Ptr(new pcl::search::KdTree<PointTC>));
        feature_extractor.setSearchSurface(filtered_pcl_pc.at(idx));
        feature_extractor.setInputCloud(keypoints.at(idx));
        feature_extractor.setInputNormals(filtered_normals.at(idx));
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature(new pcl::PointCloud<pcl::FPFHSignature33>);
        feature_extractor.compute(*feature);
        features.emplace_back(feature);
    }

    for (auto idx = 0; idx < features.size(); ++idx) {
        PointCloudCPtr kpts(new PointCloudC());
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature(new pcl::PointCloud<pcl::FPFHSignature33>());
        for (auto i = 0; i < features.at(idx)->size(); i++) {
            if (!std::isnan(features.at(idx)->points[i].histogram[0])) {
                kpts->points.push_back(keypoints.at(idx)->points[i]);
                feature->points.push_back(features.at(idx)->points[i]);
            }
        }
        keypoints[idx] = kpts;
        features[idx] = feature;
    }
    
    // find correspondences
    std::vector<pcl::CorrespondencesPtr> correspondences;
    for (auto i = 0, j = 1; i < features.size() - 1; ++i, ++j) {
        const pcl::PointCloud<pcl::FPFHSignature33>::Ptr source = features.at(i);
        const pcl::PointCloud<pcl::FPFHSignature33>::Ptr target = features.at(j);
        pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> estimator_corr;
        estimator_corr.setInputSource(source);
        estimator_corr.setInputTarget(target);
        pcl::CorrespondencesPtr match_correspondences(new pcl::Correspondences());
        estimator_corr.determineReciprocalCorrespondences(*match_correspondences);
        correspondences.emplace_back(match_correspondences);
    }

    // filter correspondences
    pcl::registration::CorrespondenceRejectorSampleConsensus<PointTC> rejector;
    rejector.setInlierThreshold(pcl_desc_cfg.inlier_threshold);
    for (auto i = 0, j = 1; i < keypoints.size() - 1; ++i, ++j) {
        const auto source = keypoints.at(i);
        const auto target = keypoints.at(j);
        rejector.setInputSource(source);
        rejector.setInputTarget(target);
        rejector.setInputCorrespondences(correspondences.at(i));
        pcl::CorrespondencesPtr good_correspondences(new pcl::Correspondences());
        rejector.getRemainingCorrespondences(*correspondences.at(i), *good_correspondences);
        rejector.getCorrespondences(*good_correspondences);
        correspondences[i] = good_correspondences;
    }

    pcl::registration::TransformationEstimationSVD<PointTC, PointTC> trans_est;
    std::vector<Mat4f> transforms;
    for (auto i = 0, j = 1; i < keypoints.size() - 1; ++i, ++j) {
        Mat4f transform;
        const auto source = keypoints.at(i);
        const auto target = keypoints.at(j);
        LOG(ERROR) << correspondences.at(i)->size();
        trans_est.estimateRigidTransformation(*source, *target, *correspondences.at(i), transform);
        transforms.emplace_back(transform);
    }

    
    
    pcl::IterativeClosestPoint<PointTC, PointTC> icp;
    icp.setMaxCorrespondenceDistance(0.7);
    icp.setTransformationEpsilon(1e-6);
    icp.setMaximumIterations(50);

    auto source_pc = filtered_pcl_pc.front();
    PointCloudCPtr align_data(new PointCloudC);
    LOG(ERROR) << "icp transform calc";
    VoxelFilterConfig vfc_icp;
    vfc_icp.resolution = 0.07;
    for (auto idx = 0; idx < transforms.size(); ++idx) {
        icp.setInputSource(source_pc);
        const auto target_pc = filtered_pcl_pc.at(idx + 1);
        icp.setInputTarget(target_pc);
        
        icp.align(*align_data, transforms.at(idx));

        transforms.at(idx) = icp.getFinalTransformation();

        source_pc = align_data;
        (*source_pc) += *target_pc;
        source_pc = voxel_filter_pc(source_pc, vfc_icp);
    }

    
    PointCloudCPtr total_pc(new PointCloudC);
    total_pc = pcl_pc.front();
    LOG(ERROR) << "concat point cloud";
    for (auto idx = 0; idx < transforms.size(); ++idx) {
        Mat4 T = transforms.at(idx).cast<r64>();
        total_pc = transform_point_cloud(total_pc, T);
        (*total_pc) += *pcl_pc.at(idx + 1);
    }

    StatisticalFilterConfig sfc;
    VoxelFilterConfig vfc_total;
    vfc_total.resolution = 0.08;
    total_pc = voxel_filter_pc(total_pc, vfc_total);
    total_pc = statistical_filter_pc(total_pc, sfc);

    return total_pc;
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


};