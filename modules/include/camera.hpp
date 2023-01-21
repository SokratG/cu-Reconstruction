#ifndef CUPHOTO_LIB_CAMERA_HPP
#define CUPHOTO_LIB_CAMERA_HPP

#include "types.hpp"
#include <opencv2/core/core.hpp>
#include <memory>

namespace cuphoto {

struct Distortion {
    Distortion() : k1(0), k2(0), p1(0), p2(0), k3(0), alpha(1.0) {}
    Distortion(const r64 k1, const r64 k2, const r64 p1, const r64 p2, const r64 k3);
    Vec5 D() const;
    r64 k1, k2, p1, p2, k3;
    r64 alpha; // free scaling parameter
};


class Camera {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<Camera>;

    Camera() : _fx(0), _fy(0), _cx(0), _cy(0) {}

    Camera(const r64 fx, const r64 fy, const r64 cx, const r64 cy);

    r64 fx() const;
    r64 fy() const;
    r64 cx() const;
    r64 cy() const;
    void fx(const r64 fx);
    void fy(const r64 fy);
    void cx(const r64 cx);
    void cy(const r64 cy);

    SE3 pose() const;
    void pose(const SE3& pose);
    SE3 inv_pose() const;

    // return intrinsic matrix
    Mat33 K() const;

    // coordinate transform: world, camera, pixel
    Vec3 world2camera(const Vec3& p_w, const SE3& T_C_W) const;

    Vec3 camera2world(const Vec3& p_c, const SE3& T_C_W) const;

    Vec2 camera2pixel(const Vec3& p_c) const;

    Vec3 pixel2camera(const Vec2& p_px, const r64 depth = 1) const;

    Vec3 pixel2world(const Vec2& p_px, const SE3& T_C_W, const r64 depth = 1) const;

    Vec2 world2pixel(const Vec3& p_w, const SE3& T_C_W) const;
 
private:
    r64 _fx, _fy, _cx, _cy;
    SE3 camera_pose;
    SE3 camera_pose_inv;
};


std::pair<Vec3, Vec3> project_px_point(const Camera::Ptr camera, const cv::Point2d src, const cv::Point2d dst);

};

#endif // CUPHOTO_CAMERA_HPP