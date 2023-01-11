#ifndef CUREC_LIB_KEYFRAME_HPP
#define CUREC_LIB_KEYFRAME_HPP

#include "types.hpp"
#include <opencv2/core/core.hpp>
#include <memory>
#include <mutex>
#include <vector>

namespace curec {

class Landmark;

class KeyFrame : public std::enable_shared_from_this<KeyFrame> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<KeyFrame>;

    // factory build mode, assign id
    static std::shared_ptr<KeyFrame> create_keyframe(const cv::Mat& image, const SE3& camera_pose);
    static std::shared_ptr<KeyFrame> create_keyframe();


    std::shared_ptr<KeyFrame> getptr() {
        return shared_from_this();
    }
    // set and get pose, thread safe
    SE3 pose() const;
    void pose(const SE3& camera_pose);
    void pose(const Mat3& rotation, const Vec3& translation);
    cv::Mat frame() const {return frame_image;}
    void frame(const cv::Mat frame) {frame_image = frame;}
    Mat34 get_projection_mat(const Mat3& K) const;

public:
    // frame id
    uuid id;
    // Timestamp
    time_point time_stamp;
protected:
    KeyFrame();
    KeyFrame(const uuid& id,
             const cv::Mat& image,
             const SE3& camera_pose,
             const time_point time_stamp);
private:
    // T_CAMERA_WORLD form Pose
    SE3 camera_pose;
    // image
    cv::Mat frame_image;
    mutable std::mutex pose_mutex;
    std::vector<std::shared_ptr<Landmark>> features;
};

};


#endif // CUREC_KEYFRAME_HPP