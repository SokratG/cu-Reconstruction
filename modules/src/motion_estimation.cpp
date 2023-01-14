#include "motion_estimation.hpp"
#include "feature_matcher.hpp"
#include "bundle_adjustment.hpp"
#include "utils.hpp"

#include <utility>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <glog/logging.h>


namespace curec {

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
    SE3 relative_motion = frames.at(ma.front().src_idx)->pose();

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
        const auto current_pose = frames[dst_idx]->pose();
        frames[dst_idx]->pose(relative_motion * current_pose);
        relative_motion = frames[dst_idx]->pose() * frames[src_idx]->pose().inverse();
    }
    return true;
}

bool MotionEstimationOptimization::estimate_motion(VisibilityGraph& landmarks,
                                                   std::vector<KeyFrame::Ptr>& frames,
                                                   const Camera::Ptr camera) {
    if (landmarks.empty() || frames.empty()) {
        LOG(ERROR) << "The given data is empty:";
        LOG(ERROR) << "frames: " << frames.empty();
        LOG(ERROR) << "landmarks: " << landmarks.empty();
        return false;
    }
    BundleAdjustment::Ptr ba = std::make_shared<BundleAdjustment>(OptimizerType::BA_CERES, 
                                                                  TypeReprojectionError::REPROJECTION_RT);
    
    ba->build_problem(landmarks, frames, camera);

    ba->solve(landmarks, frames);

    return true;
}

};