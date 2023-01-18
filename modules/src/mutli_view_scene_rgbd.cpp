#include "multi_view_scene.hpp"

#include <glog/logging.h>


namespace curec {


MultiViewSceneRGBD::MultiViewSceneRGBD(const Camera::Ptr _camera) : MultiViewScene(_camera) {

}

bool MultiViewSceneRGBD::add_frame(const cv::cuda::GpuMat rgb, const cv::cuda::GpuMat depth) {
    if (rgb.empty() || depth.empty()) {
        LOG(WARNING) << "The given image in MultiViewSceneRGBD is empty!";
        return false;
    }
    KeyFrame::Ptr kframe_rgb_ptr = KeyFrame::create_keyframe();
    kframe_rgb_ptr->frame(rgb);
    rgb_frames.emplace_back(kframe_rgb_ptr);

    KeyFrame::Ptr kframe_depth_ptr = KeyFrame::create_keyframe();
    kframe_depth_ptr->frame(depth);
    depth_frames.emplace_back(kframe_depth_ptr);

    return true;
}


void MultiViewSceneRGBD::reconstruct_scene() {

}


};