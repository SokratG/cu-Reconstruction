#include "camera.hpp"

namespace curec {

Distortion::Distortion(const r64 _k1, const r64 _k2, const r64 _p1, const r64 _p2, const r64 _k3) :
                       k1(_k1), k2(_k2), p1(_p1), p2(_p2), k3(_k3), alpha(1.0) {

}

Vec5 Distortion::D() const {
    Vec5 D;
    D << k1, k2, p1, p2, k3;
    return D;
} 

Camera::Camera(const r64 fx, const r64 fy, const r64 cx, const r64 cy) :
              _fx(fx), _fy(fy), _cx(cx), _cy(cy) {
}

r64 Camera::fx() const {
    return _fx;
}
r64 Camera::fy() const {
    return _fy;
}
r64 Camera::cx() const {
    return _cx;
}
r64 Camera::cy() const {
    return _cy;
}
void Camera::fx(const r64 fx) {
    _fx = fx;
}
void Camera::fy(const r64 fy) {
    _fy = fy;
}
void Camera::cx(const r64 cx) {
    _cx = cx;
}
void Camera::cy(const r64 cy) {
    _cy = cy;
}


SE3 Camera::pose() const {
    return camera_pose;
}

void Camera::pose(const SE3& pose) {
    camera_pose = pose;
}

SE3 Camera::inv_pose() const {
    return camera_pose.inverse();
}

Mat33 Camera::K() const {
    Mat33 K;
    K << fx(), 0, cx(), 0, fy(), cy(), 0, 0, 1;
    return K;
}


Vec3 Camera::world2camera(const Vec3& pt_world, const SE3& T_C_W) const {
    return pose() * T_C_W * pt_world;
}

Vec3 Camera::camera2world(const Vec3& pt_camera, const SE3& T_C_W) const {
    return T_C_W * inv_pose() * pt_camera;
}

Vec2 Camera::camera2pixel(const Vec3& pt_camera) const {
    return Vec2(
        fx() * pt_camera(0, 0) / pt_camera(2, 0) + cx(),
        fy() * pt_camera(1, 0) / pt_camera(2, 0) + cy()
    );
}

Vec3 Camera::pixel2camera(const Vec2& pt_px, const r64 depth) const {
    return Vec3(
        (pt_px(0, 0) - cx()) * depth / fx(),
        (pt_px(1, 0) - cy()) * depth / fy(),
        depth
    );
}

Vec3 Camera::pixel2world(const Vec2& pt_px, const SE3& T_C_W, const r64 depth) const {
    return camera2world(pixel2camera(pt_px, depth), T_C_W);
}

Vec2 Camera::world2pixel(const Vec3& pt_world, const SE3& T_C_W) const {
    return camera2pixel(world2camera(pt_world, T_C_W));
}

};