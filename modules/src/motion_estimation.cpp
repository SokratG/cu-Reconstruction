#include "motion_estimation.hpp"
#include "feature_matcher.hpp"
#include "bundle_adjustment.hpp"
#include "cp_exception.hpp"
#include "utils.hpp"

#include <utility>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <glog/logging.h>


namespace cuphoto {


BundleAdjustment::Ptr get_ba_problem(const TypeMotion tm) 
{
    switch (tm) {
        case TypeMotion::POSE_POINT:
            return std::make_shared<BundleAdjustment>(OptimizerType::BA_CERES, 
                                                      TypeReprojectionError::REPROJECTION_POSE_POINT);
        case TypeMotion::POSE:
            return std::make_shared<BundleAdjustment>(OptimizerType::BA_CERES, 
                                                      TypeReprojectionError::REPROJECTION_POSE);
        case TypeMotion::POSE_ICP:
            return std::make_shared<BundleAdjustment>(OptimizerType::BA_CERES_ICP,
                                                      TypeReprojectionError::REPROJECTION_POSE);
        default:
            throw CuPhotoException("Error: Unknown type of motion in optimization!");
    }
}


void MotionEstimationRansac::estimate_ransac(const std::vector<cv::Point2d>& src, 
                                             const std::vector<cv::Point2d>& dst,
                                             const cv::Mat K,
                                             Mat3& R, Vec3& t) {
    cv::Mat E = cv::findEssentialMat(src, dst, K, cv::FM_RANSAC, 0.9, 1.5);
    cv::Mat cvR, cvt;
    cv::recoverPose(E, src, dst, K, cvR, cvt);
    cv::cv2eigen(cvR, R); cv::cv2eigen(cvt, t);
}

bool MotionEstimationRansac::estimate_motion(std::vector<KeyFrame::Ptr>& frames,
                                             const std::vector<MatchAdjacent>& ma,
                                             const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                             const Camera::Ptr camera) {
    if (frames.empty() || ma.empty() || feat_pts.empty()) {
        LOG(ERROR) << "The given data is empty:";
        LOG(ERROR) << "frames:" << frames.empty();
        LOG(ERROR) << "matching:" << ma.empty();
        LOG(ERROR) << "feature points:" << feat_pts.empty();
        return false;
    }
    cv::Mat K;
    cv::eigen2cv(camera->K(), K);

    for (const auto& adj : ma) {
        const i32 src_idx = adj.src_idx;
        if (adj.dst_idx < 0)
            continue;
        const i32 dst_idx = adj.dst_idx;
        const auto& match = adj.match;
        const auto match_size = match.size();

        std::vector<cv::Point2d> src(match_size), dst(match_size);
        for (i32 i = 0; i < match_size; ++i) {
            src[i] = feat_pts[src_idx].at(match.at(i).queryIdx)->position.pt;
            dst[i] = feat_pts[dst_idx].at(match.at(i).trainIdx)->position.pt;
        }
        Mat3 R; Vec3 t;
        estimate_ransac(src, dst, K, R, t);
        frames[dst_idx]->pose(R, t);
        const auto src2dst_pose = frames[dst_idx]->pose();
        frames[dst_idx]->pose(frames[src_idx]->pose() * src2dst_pose);
    }
    return true;
}


bool MotionEstimationICP::estimate_motion(std::vector<KeyFrame::Ptr>& frames,
                                          const std::vector<MatchAdjacent>& ma,
                                          const std::unordered_map<i32, ConnectionPoints>& pts3D) {
    if (pts3D.empty() || ma.empty() || frames.empty()) {
        LOG(ERROR) << "The given data is empty:";
        LOG(ERROR) << "frames: " << frames.empty();
        LOG(ERROR) << "matching:" << ma.empty();
        LOG(ERROR) << "3D points : " << pts3D.empty();
        return false;
    }

    for (const auto& adj : ma) {
        const i32 src_idx = adj.src_idx;
        if (adj.dst_idx < 0)
            continue;
        const i32 dst_idx = adj.dst_idx;
        const auto& pts = pts3D.at(src_idx);

        Mat3 R; Vec3 t;
        estimate_icp(pts, R, t);

        frames[dst_idx]->pose(R, t);
        const auto src2dst_pose = frames[dst_idx]->pose();
        frames[dst_idx]->pose(frames[src_idx]->pose() * src2dst_pose);
    }

    return true;
}


void MotionEstimationICP::estimate_icp(const ConnectionPoints& pts, Mat3& R, Vec3& t) {
    Vec3 center_p1, center_p2; // center mass
    const auto N = pts.size();
    for (auto i = 0; i < N; ++i) {
        center_p1 += pts[i].first;
        center_p2 += pts[i].second;
    }

    center_p1 = center_p1 / N;
    center_p2 = center_p2 / N;

    // remove the center - de-centroid coordinates
    std::vector<Vec3> q1(N), q2(N);
    for (auto i = 0; i < N; ++i) {
        q1[i] = pts[i].first - center_p1;
        q2[i] = pts[i].second - center_p2;
    }

    // q1∗q2^T
    Mat3 W = Mat3::Zero();
    for (auto i = 0; i < N; ++i)
        W += q1[i] * q2[i].transpose();
    
    Eigen::JacobiSVD<Mat3> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 U = svd.matrixU();
    Mat3 V = svd.matrixV();
    R = U * (V.transpose());
    if (R.determinant() < 0)
        R = -R;

    t = center_p1 - R * center_p2;
}


bool MotionEstimationOptimization::estimate_motion(std::vector<Landmark::Ptr>& landmarks,
                                                   std::vector<KeyFrame::Ptr>& frames,
                                                   const VisibilityGraph& vis_graph,
                                                   const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                                   const Camera::Ptr camera,
                                                   const TypeMotion tm) {
    if (landmarks.empty() || frames.empty()) {
        LOG(ERROR) << "The given data is empty:";
        LOG(ERROR) << "frames: " << frames.empty();
        LOG(ERROR) << "landmarks: " << landmarks.empty();
        return false;
    }
    BundleAdjustment::Ptr ba = get_ba_problem(tm);
    
    ba->build_problem(vis_graph, landmarks, frames, feat_pts, camera);

    ba->solve(landmarks, frames);

    return true;
}

bool MotionEstimationOptimization::estimate_motion(std::vector<KeyFrame::Ptr>& frames,
                                                   const std::vector<MatchAdjacent>& ma,
                                                   const std::unordered_map<i32, ConnectionPoints>& pts3D,
                                                   const TypeMotion tm) {
    if (pts3D.empty() || ma.empty() || frames.empty()) {
        LOG(ERROR) << "The given data is empty:";
        LOG(ERROR) << "frames: " << frames.empty();
        LOG(ERROR) << "matching:" << ma.empty();
        LOG(ERROR) << "3D points : " << pts3D.empty();
        return false;
    }

    BundleAdjustment::Ptr ba = get_ba_problem(tm);

    ba->build_problem(pts3D, ma, frames);

    ba->solve(frames);

    return true;
}


bool triangulation(const SE3& src_pose,
                   const SE3& dst_pose,
                   const std::pair<Vec3, Vec3>& points,
                   const r64 confidence_thrshold,
                   Vec3 &pt_world) {
    MatXX A(4, 4);
    VecX b(4);
    b.setZero();
    Mat34 m = src_pose.matrix3x4();
    A.block<1, 4>(0, 0) = points.first[0] * m.row(2) - m.row(0);
    A.block<1, 4>(1, 0) = points.first[1] * m.row(2) - m.row(1); 
    m = dst_pose.matrix3x4();
    A.block<1, 4>(2, 0) = points.second[0] * m.row(2) - m.row(0);
    A.block<1, 4>(3, 0) = points.second[1] * m.row(2) - m.row(1);

    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

    if (svd.singularValues()[3] / svd.singularValues()[2] < confidence_thrshold) {
        return true;
    }
    return false;
}


};