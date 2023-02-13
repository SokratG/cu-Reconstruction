#include "multi_view_scene_rgbd.hpp"
#include "visibility_graph.hpp"
#include "motion_estimation.hpp"

#include "cp_exception.hpp"
#include "point_cloud_utility.hpp"

#include <glog/logging.h>



namespace cuphoto {


MultiViewSceneRGBD::MultiViewSceneRGBD(const Camera::Ptr _camera, const Camera::Ptr _depth_camera) : 
                                       MultiViewScene(_camera), depth_camera(_depth_camera) {

}

bool MultiViewSceneRGBD::add_frame(const cv::cuda::GpuMat rgb, const cv::cuda::GpuMat depth) {
    if (rgb.empty() || depth.empty()) {
        LOG(WARNING) << "The given image in MultiViewSceneRGBD is empty!";
        return false;
    }
    KeyFrame::Ptr kframe_rgb_ptr = KeyFrame::create_keyframe();
    kframe_rgb_ptr->frame(rgb);
    KeyFrame::Ptr kframe_depth_ptr = KeyFrame::create_keyframe();
    kframe_depth_ptr->frame(depth);

    rgbd_frames.emplace_back(std::make_shared<RGBD>(kframe_rgb_ptr, kframe_depth_ptr));
    return true;
}


void MultiViewSceneRGBD::reconstruct_scene() {
    if (rgbd_frames.empty()) {
        throw CuPhotoException("The image data is empty! Can't run SFM pipeline");
    }

    estimate_motion();

    const auto width = rgbd_frames.front()->depth->frame().cols;
    const auto height = rgbd_frames.front()->depth->frame().rows;
    const size_t total_number_pts = rgbd_frames.size() * width * height;
    
    // cuda_pc = cudaPointCloud::create(K, total_number_pts);

    for (auto idx = 0; idx < rgbd_frames.size(); ++idx) {
        const KeyFrame::Ptr depth = rgbd_frames.at(idx)->depth;
        const KeyFrame::Ptr rgb = rgbd_frames.at(idx)->rgb;

        cudaPointCloud::Ptr tmp_cu_pc = pcl_to_cuda_pc(rgb, depth, camera);
        
    }
    
    // cuda_pc->filter_depth(0.0f);

    // statistical_filter_pc(cuda_pc);
}


std::tuple<std::vector<MultiViewSceneRGBD::RGB>, std::vector<cv::Mat>> MultiViewSceneRGBD::split_rgbd() {
    std::vector<RGB> rgb_frames(rgbd_frames.size());
    std::vector<cv::Mat> depth_frames(rgbd_frames.size());
    for (auto i = 0; i < rgb_frames.size(); ++i) {
        rgb_frames[i] = rgbd_frames[i]->rgb;
        rgbd_frames[i]->depth->frame().download(depth_frames[i]);
    }
    return {rgb_frames, depth_frames};
}

void MultiViewSceneRGBD::filter_outlier_frames(std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                               std::vector<cv::cuda::GpuMat>& descriptors) {
    auto [rgb_frames, depth_frames] = split_rgbd();

    detect_feature(rgb_frames, feat_pts, descriptors);
    std::vector<MatchAdjacent> matching;
    matching_feature(feat_pts, descriptors, matching);

    std::vector<RGBD::Ptr> filtered_rgbd_frames;
    std::vector<cv::cuda::GpuMat> filtered_descriptors;
    std::vector<std::vector<Feature::Ptr>> filtered_feature_pts;
    for (const auto& match : matching) {
        const auto src_idx = match.src_idx;
        if (match.dst_idx != -1) {
            filtered_descriptors.emplace_back(descriptors.at(src_idx));
            filtered_feature_pts.emplace_back(feat_pts.at(src_idx));
            filtered_rgbd_frames.emplace_back(rgbd_frames.at(src_idx));
        }
    }
    descriptors = filtered_descriptors;
    feat_pts = filtered_feature_pts;
    rgbd_frames = filtered_rgbd_frames;
}

void MultiViewSceneRGBD::estimate_motion() {
    std::vector<std::vector<Feature::Ptr>> feat_pts;
    std::vector<cv::cuda::GpuMat> descriptors;

    LOG(INFO) << "Filter given images by outlier features";
    filter_outlier_frames(feat_pts, descriptors);

    if (rgbd_frames.size() < 2) {
        throw CuPhotoException("Not enough images for reconstruction pipeline! A lot of outliers.");
    }

    LOG(INFO) << "Current number of images after filtering is: " << rgbd_frames.size();
    
    auto [rgb_frames, depth_frames] = split_rgbd();

    std::vector<MatchAdjacent> matching;
    matching_feature(feat_pts, descriptors, matching);

    std::unordered_map<i32, ConnectionPoints> conn_pts;
    build_visibility_connection_points(matching, rgb_frames, depth_frames, feat_pts, camera, conn_pts);

    MotionEstimationICP::Ptr me_icp = std::make_shared<MotionEstimationICP>();
    me_icp->estimate_motion(rgb_frames, matching, conn_pts);

    MotionEstimationOptimization::Ptr me_optim = std::make_shared<MotionEstimationOptimization>();
    me_optim->estimate_motion(rgb_frames, matching, conn_pts);

    for (auto i = 0; i < rgb_frames.size(); ++i) {
        rgbd_frames[i]->rgb->pose(rgb_frames[i]->pose());
        rgbd_frames[i]->depth->pose(rgb_frames[i]->pose()); // ? depth camera parameters
    }
}


};