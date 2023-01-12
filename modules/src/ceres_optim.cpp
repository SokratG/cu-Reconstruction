#include "ceres_optim.hpp"

namespace curec {

CeresCameraModel::CeresCameraModel(const SE3& camera_pose, const Mat3& K) {
    Mat3 R = camera_pose.rotationMatrix();
    Vec3 t = camera_pose.translation();
    Eigen::Quaterniond q(R);
    raw_camera_param[0] = q.w();
    raw_camera_param[1] = q.x();
    raw_camera_param[2] = q.y();
    raw_camera_param[3] = q.z();
    raw_camera_param[4] = t.x();
    raw_camera_param[5] = t.y();
    raw_camera_param[6] = t.z();
    raw_camera_param[7] = K(0, 0);
    raw_camera_param[8] = K(1, 1);
    camera_center = Vec2(K(0, 2), K(2, 2));
}

Mat3 CeresCameraModel::K() const {
    Mat3 K;
    K << raw_camera_param[7], 0, camera_center.x(), 0, raw_camera_param[8], camera_center.y(), 0, 0, 1;
    return K;
}

SE3 CeresCameraModel::pose() const {
    Eigen::Quaterniond q(raw_camera_param[0], raw_camera_param[1], raw_camera_param[2], raw_camera_param[3]);
    Vec3 t(raw_camera_param[4], raw_camera_param[5], raw_camera_param[6]);
    SE3 cam_pose(q, t);
    return cam_pose;
}


};
