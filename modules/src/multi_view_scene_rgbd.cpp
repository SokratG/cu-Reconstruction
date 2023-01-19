#include "multi_view_scene_rgbd.hpp"
#include "visibility_graph.hpp"
#include "motion_estimation.hpp"

#include "cr_exception.hpp"

#include <glog/logging.h>


namespace curec {


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
        throw CuRecException("The image data is empty! Can't run SFM pipeline");
    }

    estimate_motion();

    // TODO
}


void MultiViewSceneRGBD::estimate_motion() {
    std::vector<RGB> rgb_frames(rgbd_frames.size());
    std::vector<cv::Mat> depth_frames(rgbd_frames.size());
    for (auto i = 0; i < rgb_frames.size(); ++i) {
        rgb_frames[i] = rgbd_frames[i]->rgb;
        rgbd_frames[i]->depth->frame().download(depth_frames[i]);
    }

    std::vector<std::vector<Feature::Ptr>> feat_pts;
    std::vector<MatchAdjacent> matching;
    detect_feature(rgb_frames, feat_pts, matching);

    VisibilityGraph vis_graph;
    std::vector<Landmark::Ptr> landmarks;
    build_landmarks_graph_depth(matching, rgb_frames, depth_frames, feat_pts, camera, vis_graph, landmarks);
    
    MotionEstimationOptimization::Ptr me_optim = std::make_shared<MotionEstimationOptimization>();

    me_optim->estimate_motion(landmarks, rgb_frames, vis_graph, feat_pts, camera);

    for (auto i = 0; i < rgb_frames.size(); ++i) {
        rgbd_frames[i]->rgb->pose(rgb_frames[i]->pose());
        rgbd_frames[i]->depth->pose(rgb_frames[i]->pose()); // ? with depth camera parameters
    }
}


};