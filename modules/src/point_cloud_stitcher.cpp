#include "point_cloud_stitcher.hpp"
#include "point_cloud_utility.hpp"

#include "cp_exception.hpp"

#include <pcl/filters/extract_indices.h>

#include <pcl/keypoints/sift_keypoint.h>

#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>


namespace cuphoto {

struct PCLMatchAdjacent {
    i32 src_idx;
    i32 dst_idx;
    std::vector<i32> correspondences;
    PCLMatchAdjacent() = delete;
    PCLMatchAdjacent(const i32 _src_idx) : src_idx(_src_idx), dst_idx(-1) {}
};


PointCloudStitcher::PointCloudStitcher(const PointCloudStitcherBackend _pcsb) : pcsb(_pcsb) {

}

PointCloudCPtr PointCloudStitcher::stitch(const std::vector<PointCloudCPtr>& pcl_pc,
                                          std::vector<Mat4>& transforms,
                                          const PointCloudStitcherConfig& pcsc) {
    std::vector<Mat4f> transformsf = std::vector<Mat4f>();
    switch(pcsb) {
        case PointCloudStitcherBackend::FEATURE_ESTIMATOR_LM:
            transform_estimation_rigid_lm(pcl_pc, transformsf, pcsc);
            break;
        case PointCloudStitcherBackend::ICP_ESTIMATOR:
            transform_estimation_icp(pcl_pc, transformsf, pcsc);
            break;
        case PointCloudStitcherBackend::FEATURE_AND_ICP_ESTIMATOR:
            transform_estimation_rigid_lm(pcl_pc, transformsf, pcsc);
            transform_estimation_icp(pcl_pc, transformsf, pcsc);
            break;
        default:
            throw CuPhotoException("The given point cloud stitcher backend is not allowed!");
    }

    for (auto idx = 0; idx < transformsf.size(); ++idx) {
        transforms.emplace_back(transformsf.at(idx).cast<r64>());
    }

    const auto stitched_pc = stitch_by_transformation(pcl_pc, transforms);


    return stitched_pc;
}

void PointCloudStitcher::transform_estimation_icp(const std::vector<PointCloudCPtr>& pcl_pc,
                                                  std::vector<Mat4f>& transforms,
                                                  const PointCloudStitcherConfig& pcsc) {
    if (transforms.empty()) {
        transforms = std::vector<Mat4f>(pcl_pc.size() - 1);
        for (auto idx = 0; idx < transforms.size(); ++idx)
            transforms.at(idx) = Mat4f::Identity();
    }
    
    VoxelFilterConfig vfc_icp;
    VoxelFilterConfig vfc_icp_step;
    vfc_icp.resolution = pcsc.icp_resolution_voxel_grid;
    vfc_icp_step.resolution = pcsc.icp_step_resolution_point_cloud;

    pcl::IterativeClosestPoint<PointTC, PointTC> icp;
    icp.setMaxCorrespondenceDistance(pcsc.icp_max_correspond_dist);
    icp.setTransformationEpsilon(pcsc.icp_transformation_eps);
    icp.setMaximumIterations(pcsc.icp_max_iteration);
    icp.setRANSACOutlierRejectionThreshold(pcsc.icp_ransac_threshold);

    auto source_pc = voxel_filter_pc(pcl_pc.front(), vfc_icp);
    PointCloudCPtr align_data(new PointCloudC);

    for (auto idx = 0; idx < transforms.size(); ++idx) {
        icp.setInputSource(source_pc);
        const auto target_pc = voxel_filter_pc(pcl_pc.at(idx + 1), vfc_icp);
        icp.setInputTarget(target_pc);
        
        icp.align(*align_data, transforms.at(idx));

        transforms.at(idx) = icp.getFinalTransformation();

        source_pc = align_data;
        (*source_pc) += *target_pc;
        source_pc = voxel_filter_pc(source_pc, vfc_icp_step);
    }
}

void PointCloudStitcher::transform_estimation_rigid_lm(const std::vector<PointCloudCPtr>& pcl_pc,
                                                       std::vector<Mat4f>& transforms,
                                                       const PointCloudStitcherConfig& pcsc) {
    // computer normals
    std::vector<PointCloudCPtr> filtered_pcl_pc(pcl_pc.size());
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> filtered_normals(pcl_pc.size());
    for (auto idx = 0; idx < pcl_pc.size(); ++idx) {
        pcl::NormalEstimationOMP<PointTC, pcl::Normal> normal_estimation;
        normal_estimation.setSearchMethod(pcl::search::Search<PointTC>::Ptr(new pcl::search::KdTree<PointTC>));
        normal_estimation.setRadiusSearch(pcsc.desc_normal_radius_search);
        normal_estimation.setInputCloud(pcl_pc.at(idx));

        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        normal_estimation.compute(*normals);
        PointCloudCPtr filter_pc(new PointCloudC);
        pcl::PointCloud<pcl::Normal>::Ptr filter_normals(new pcl::PointCloud<pcl::Normal>);
        pcl::PointIndices::Ptr pt_inds(new pcl::PointIndices());
        pcl::removeNaNNormalsFromPointCloud(*normals, *filter_normals, pt_inds->indices);
        pcl::ExtractIndices<PointTC> extract;
        extract.setInputCloud(pcl_pc.at(idx));
        extract.setIndices(pt_inds);
        extract.filter(*filter_pc);

        filtered_pcl_pc.at(idx) = filter_pc;
        filtered_normals.at(idx) = filter_normals;
    }

    // keypoint detection
    pcl::SIFTKeypoint<PointTC, PointTC> sift;
    sift.setScales(pcsc.sift_min_scale, pcsc.sift_n_octaves, pcsc.sift_n_scales_per_octave);
    sift.setMinimumContrast(pcsc.sift_min_contrast);
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
    feature_extractor.setRadiusSearch(pcsc.desc_feature_radius_search);
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


    // filter NaN descriptor
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
    rejector.setInlierThreshold(pcsc.desc_inlier_threshold);
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
    for (auto i = 0, j = 1; i < keypoints.size() - 1; ++i, ++j) {
        Mat4f transform = Mat4f::Identity();
        const auto source = keypoints.at(i);
        const auto target = keypoints.at(j);
        trans_est.estimateRigidTransformation(*source, *target, *correspondences.at(i), transform);
        transforms.emplace_back(transform);
    }
}

PointCloudCPtr PointCloudStitcher::stitch_by_transformation(const std::vector<PointCloudCPtr>& pcl_pc,
                                                            const std::vector<Mat4>& transforms) {
    PointCloudCPtr total_pc(new PointCloudC);
    (*total_pc) += (*pcl_pc.front());
    for (auto idx = 0; idx < transforms.size(); ++idx) {
        const Mat4 T = transforms.at(idx);
        total_pc = transform_point_cloud(total_pc, T);
        (*total_pc) += *pcl_pc.at(idx + 1);
    }
    return total_pc;
}


};
