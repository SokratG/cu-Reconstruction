#include "sfm.hpp"
#include "feature_detector.hpp"
#include "feature_matcher.hpp"
#include "motion_estimation.hpp"
#include "point_cloud.hpp"
#include "utils.hpp"
#include <glog/logging.h>


namespace curec {

Sfm::Sfm(const Camera::Ptr _camera) : camera(_camera) {

}

bool Sfm::add_frame(const cv::Mat frame) {
    if (frame.empty()) {
        LOG(WARNING) << "The given image in SFM is empty!";
        return false;
    }
    KeyFrame::Ptr kframe_ptr = KeyFrame::create_keyframe();
    kframe_ptr->frame(frame);
    frames.emplace_back(kframe_ptr);
    return true;
}

void Sfm::build_landmark_graph() {
    FeatureDetector fd(FeatureDetectorBackend::SIFT, std::string());
    std::vector<cv::Mat> descriptors;
    std::vector<std::vector<Feature::Ptr>> feat_pts;
    for (const auto frame : frames) {
        std::vector<Feature::Ptr> kpts;
        cv::Mat descriptor;
        fd.detect(frame, kpts, descriptor);
        descriptors.emplace_back(descriptor);
        feat_pts.emplace_back(kpts);
    }

    std::vector<MatchAdjacent> matching = feature_matching(FeatureMatcherBackend::BRUTEFORCE,
                                                           descriptors);
    
    LOG(INFO) << "matching size: " << matching.size();
    
    
    matching = ransac_filter_outlier(matching, feat_pts, camera->K());
    LOG(INFO) << "matching size after filter outliers: " << matching.size();

    MotionEstimation::Ptr me = std::make_shared<MotionEstimation>();
    MotionEstimation::VisibilityGraph landmarks;
    me->estimate_motion_ransac(frames, matching, feat_pts, camera, landmarks);

    auto size = 0;
    for (const auto& lm : landmarks) {
        size += lm.second.size();
    }
    LOG(INFO) << "Total landmark size: " << size;


    std::vector<Vec3> pts;
    std::vector<Vec3f> colors;
    std::vector<SE3> poses;
    for (i32 i = 1; i < frames.size(); ++i) {
        poses.emplace_back(frames[i]->pose());
    }
    for (const auto& cam : landmarks) {
        for (const auto& lm: cam.second) {
            const auto landmark = lm.second;
            if (landmark->pose().z() > 50.0)
                continue;
            pts.emplace_back(landmark->pose());
            colors.emplace_back(landmark->color());
        }
    }
    write_ply_file("some.ply", poses, pts, colors);

    // TODO: motion estimation (non linear optimization)
    me->estimate_motion_non_lin_opt();
}

};
