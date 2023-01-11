#include "keyframe.hpp"
#include "utils.hpp"


namespace curec {

KeyFrame::KeyFrame() {

}

KeyFrame::KeyFrame(const uuid& _id, const cv::Mat& _image,
                   const SE3& _camera_pose, const time_point _time_stamp) :
    id(_id), time_stamp(_time_stamp), camera_pose(_camera_pose), frame_image(_image) {
    
}

SE3 KeyFrame::pose() const {
    std::unique_lock<std::mutex> lck(pose_mutex);
    return camera_pose;
}

void KeyFrame::pose(const SE3& _camera_pose) {
    std::unique_lock<std::mutex> lck(pose_mutex);
    camera_pose = _camera_pose;
}

void KeyFrame::pose(const Mat3& rotation, const Vec3& translation) {
    std::unique_lock<std::mutex> lck(pose_mutex);
    SE3 se3(rotation, translation);
    camera_pose = se3;
}

KeyFrame::Ptr KeyFrame::create_keyframe(const cv::Mat& image,
                                        const SE3& camera_pose) {
    const uuid id = UUID::gen();
    const time_point time_stamp = system_clock::now();
    KeyFrame::Ptr key_frame = std::shared_ptr<KeyFrame>(new KeyFrame(id, image, camera_pose, time_stamp));
    return key_frame;
}

KeyFrame::Ptr KeyFrame::create_keyframe() {
    const uuid id = UUID::gen();
    const time_point time_stamp = system_clock::now();
    KeyFrame::Ptr key_frame = std::shared_ptr<KeyFrame>(new KeyFrame);
    key_frame->id = id;
    key_frame->time_stamp = time_stamp;
    return key_frame;
}

Mat34 KeyFrame::get_projection_mat(const Mat3& K) const {
    std::unique_lock<std::mutex> lck(pose_mutex);
    Mat3 R = camera_pose.rotationMatrix();
    Vec3 t = camera_pose.translation();
    Mat34 Rt;
    Rt.leftCols(R.cols()) = R;
    Rt.rightCols(t.cols()) = t;
    // Rt.leftCols(R.cols()) = R.transpose();
    // Rt.rightCols(t.cols()) = -R.transpose() * t;
    return K * Rt;
}

};