#include "keyframe.hpp"
#include "utils.hpp"

namespace curec {

KeyFrame::KeyFrame(const uuid& _id, const cv::Mat& _image,
                   const SE3& _camera_pose, const time_point _time_stamp) :
    id(_id), time_stamp(_time_stamp), camera_pose(_camera_pose), frame_image(_image) {
    
}

SE3 KeyFrame::pose() {
    std::unique_lock<std::mutex> lck(pose_mutex);
    return camera_pose;
}

void KeyFrame::pose(const SE3& _camera_pose) {
    std::unique_lock<std::mutex> lck(pose_mutex);
    camera_pose = _camera_pose;
}

KeyFrame::Ptr create_keyframe(const cv::Mat& image,
                              const SE3& camera_pose) {
    const uuid id = UUID::gen();
    const time_point time_stamp = system_clock::now();
    KeyFrame::Ptr key_frame = std::make_shared<KeyFrame>(id, image, camera_pose, time_stamp);
    return key_frame;
}

KeyFrame::Ptr create_keyframe() {
    const uuid id = UUID::gen();
    const time_point time_stamp = system_clock::now();
    KeyFrame::Ptr key_frame = std::make_shared<KeyFrame>();
    key_frame->id = id;
    key_frame->time_stamp = time_stamp;
    return key_frame;
}


};