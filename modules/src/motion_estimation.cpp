#include "motion_estimation.hpp"
#include "utils.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <glog/logging.h>


namespace curec {

Landmark::Ptr make_landmark(const Feature::Ptr feat_pt, const Vec3 position_world) 
{
    const auto pt2 = feat_pt->position.pt;
    const Vec3f color = cv_rgb_2_eigen_rgb(feat_pt->frame.lock()->frame().at<cv::Vec3b>(pt2));
    Landmark::Ptr landmark = Landmark::create_landmark(position_world, color);
    landmark->set_observation(feat_pt);
    return landmark;
}


bool MotionEstimation::estimate_motion_ransac(std::vector<KeyFrame::Ptr>& frames,
                                              const std::vector<MatchAdjacent>& ma,
                                              const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                              const Camera::Ptr camera,
                                              VisibilityGraph& landmarks) {
    if (frames.empty() || ma.empty() || feat_pts.empty()) {
        LOG(ERROR) << "The given data is empty:";
        LOG(ERROR) << "frames:" << frames.empty();
        LOG(ERROR) << "matching:" << ma.empty();
        LOG(ERROR) << "feature points:" << feat_pts.empty();
        return false;
    }
    cv::Mat K;
    cv::eigen2cv(camera->K(), K);
    // TODO: check require relative motion
    for (const auto& adj : ma) {
        const i32 src_idx = adj.src_idx;
        const i32 dst_idx = adj.ord_match.begin()->first;
        const auto& match = adj.ord_match.begin()->second;
        const i32 match_size = match.size();

        std::vector<cv::Point2d> src(match_size), dst(match_size);
        for (i32 i = 0; i < match_size; ++i) {
            src[i] = feat_pts[src_idx].at(match.at(i).queryIdx)->position.pt;
            dst[i] = feat_pts[dst_idx].at(match.at(i).trainIdx)->position.pt;
        }

        cv::Mat E = cv::findEssentialMat(src, dst, K, cv::FM_RANSAC, 0.9, 1.5);
        cv::Mat cvR, cvt;
        cv::recoverPose(E, src, dst, K, cvR, cvt);
        Mat3 R; Vec3 t;
        cv::cv2eigen(cvR, R); cv::cv2eigen(cvt, t);
        frames[dst_idx]->pose(R, t);
        std::vector<SE3> poses {frames[src_idx]->pose(), frames[dst_idx]->pose()};
        std::unordered_map<ui64, Landmark::Ptr> lms;
        for (i32 i = 0; i < match_size; ++i) {
            Vec3 pt_world = Vec3::Zero();
            std::vector<Vec3> pt_camera {
                camera->pixel2camera(
                    Vec2(src[i].x,
                         src[i].y)
                ),
                camera->pixel2camera(
                    Vec2(dst[i].x,
                         dst[i].y)
                ),
            };
            const bool result = triangulation(poses, pt_camera, 15e-2, pt_world);
            if (result && pt_world[2] > 0) {
                const auto pt_idx = match.at(i).trainIdx;
                auto feat_pt = feat_pts[dst_idx].at(pt_idx);
                feat_pt->frame = frames[dst_idx];
                Landmark::Ptr landmark = make_landmark(feat_pt, pt_world);
                lms.insert({pt_idx, landmark});
            }
        }
        landmarks.insert({dst_idx, lms});
    }
    return true;
}

bool MotionEstimation::estimate_motion_non_lin_opt() {
    return true;
}

};