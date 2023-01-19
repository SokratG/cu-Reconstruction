#include "sfm.hpp"
#include "feature_detector.hpp"
#include "feature_matcher.hpp"
#include "motion_estimation.hpp"
#include "utils.hpp"
#include "cr_exception.hpp"

#include <numeric>
#include <glog/logging.h>

namespace curec {


Sfm::Sfm(const Camera::Ptr _camera) : camera(_camera) {

}


bool Sfm::add_frame(const cv::cuda::GpuMat frame) {
    if (frame.empty()) {
        LOG(WARNING) << "The given image in SFM is empty!";
        return false;
    }
    KeyFrame::Ptr kframe_ptr = KeyFrame::create_keyframe();
    kframe_ptr->frame(frame);
    frames.emplace_back(kframe_ptr);
    return true;
}


std::vector<KeyFrame::Ptr> Sfm::get_frames() const {
    return frames;
}


void Sfm::run_pipeline() {
    // TODO add config struct
    if (frames.empty()) {
        throw CuRecException("The image data is empty! Can't run SFM pipeline");
    }

    std::vector<std::vector<Feature::Ptr>> feat_pts;
    std::vector<MatchAdjacent> matching;
    detect_feature(feat_pts, matching);

    LOG(INFO) << "Feature matching size: " << matching.size();

    estimation_motion(matching, feat_pts);

    LOG(INFO) << "Total landmark size: " << landmarks.size();

}


void Sfm::store_to_ply(const std::string_view& ply_filepath, const r64 range_threshold) {
    std::vector<Vec3> pts;
    std::vector<Vec3f> colors;
    std::vector<SE3> poses;
    for (i32 i = 0; i < frames.size(); ++i) {
        poses.emplace_back(frames[i]->pose());
    }

    for (const auto& landmark : landmarks) {
        if (landmark->pose().z() > range_threshold || 
            landmark->pose().y() > range_threshold ||
            landmark->pose().y() < -range_threshold ||
            landmark->pose().x() > range_threshold ||
            landmark->pose().x() < -range_threshold)
            continue;
        pts.emplace_back(landmark->pose());
        colors.emplace_back(landmark->color());
    }
    
    write_ply_file(ply_filepath, poses, pts, colors);
}


void Sfm::detect_feature(std::vector<std::vector<Feature::Ptr>>& feat_pts, std::vector<MatchAdjacent>& matching) {
    // TODO add config
    FeatureDetector fd(FeatureDetectorBackend::SIFT, "");
    std::vector<cv::cuda::GpuMat> descriptors;
    for (const auto frame : frames) {
        std::vector<Feature::Ptr> kpts;
        cv::cuda::GpuMat descriptor;
        fd.detectAndCompute(frame, kpts, descriptor);
        descriptors.emplace_back(descriptor);
        feat_pts.emplace_back(kpts);
    }

    matching = feature_matching(descriptors,
                                feat_pts,
                                camera->K());
}


void Sfm::estimation_motion(const std::vector<MatchAdjacent>& matching,
                            std::vector<std::vector<Feature::Ptr>>& feat_pts) {
    MotionEstimationRansac::Ptr me_ransac = std::make_shared<MotionEstimationRansac>();
    
    me_ransac->estimate_motion(frames, matching, feat_pts, camera);

    build_landmarks_graph_triangluation(matching, frames, feat_pts, camera, vis_graph, landmarks);

    MotionEstimationOptimization::Ptr me_optim = std::make_shared<MotionEstimationOptimization>();

    me_optim->estimate_motion(landmarks, frames, vis_graph, feat_pts, camera);

    
    for (auto lm_it = landmarks.begin(); lm_it != landmarks.end(); ) {
        if ((*lm_it)->pose().z() < 0)
            lm_it = landmarks.erase(lm_it);
        else 
            ++lm_it;
    }
    
}


};
