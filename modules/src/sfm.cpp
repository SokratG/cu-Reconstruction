#include "sfm.hpp"
#include "feature_detector.hpp"
#include "feature_matcher.hpp"
#include "motion_estimation.hpp"
#include "utils.hpp"
#include "cr_exception.hpp"

#include <numeric>
#include <glog/logging.h>


namespace curec {


static Landmark::Ptr make_landmark(const Feature::Ptr feat_pt, const Vec3& position_world) 
{
    const auto pt2 = feat_pt->position.pt;
    const Vec3f color = cv_rgb_2_eigen_rgb(feat_pt->frame.lock()->frame().at<cv::Vec3b>(pt2));
    Landmark::Ptr landmark = Landmark::create_landmark(position_world, color);
    landmark->observation(feat_pt);
    return landmark;
}


static std::pair<Vec3, Vec3> project_px_point(const Camera::Ptr camera, const cv::Point2d src, const cv::Point2d dst) {
    std::pair<Vec3, Vec3> pt_camera {
        camera->pixel2camera(
            Vec2(src.x,
                 src.y)
        ),
        camera->pixel2camera(
            Vec2(dst.x,
                 dst.y)
        ),
    };
    return pt_camera;
}


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

    i32 landmark_size = std::accumulate(landmarks.begin(), landmarks.end(), 0, 
                                        [&](const auto sum, const auto it) {
                                            return sum + it.second.size();
                                        });

    LOG(INFO) << "Total landmark size: " << landmark_size;

}


void Sfm::store_to_ply(const std::string_view& ply_filepath, const r64 range_threshold) {
    std::vector<Vec3> pts;
    std::vector<Vec3f> colors;
    std::vector<SE3> poses;
    for (i32 i = 0; i < frames.size(); ++i) {
        poses.emplace_back(frames[i]->pose());
    }
    for (const auto& cam : landmarks) {
        for (const auto& lm: cam.second) {
            const auto landmark = lm.second;
            if (landmark->pose().z() > range_threshold || 
                landmark->pose().y() > range_threshold ||
                landmark->pose().y() < -range_threshold ||
                landmark->pose().x() > range_threshold ||
                landmark->pose().x() < -range_threshold)
                continue;
            pts.emplace_back(landmark->pose());
            colors.emplace_back(landmark->color());
        }
    }
    write_ply_file(ply_filepath, poses, pts, colors);
}


void Sfm::detect_feature(std::vector<std::vector<Feature::Ptr>>& feat_pts, std::vector<MatchAdjacent>& matching) {
    // TODO add config
    FeatureDetector fd(FeatureDetectorBackend::SIFT, "");
    std::vector<cv::Mat> descriptors;
    for (const auto frame : frames) {
        std::vector<Feature::Ptr> kpts;
        cv::Mat descriptor;
        fd.detect(frame, kpts, descriptor);
        descriptors.emplace_back(descriptor);
        feat_pts.emplace_back(kpts);
    }

    matching = feature_matching(FeatureMatcherBackend::BRUTEFORCE,
                                descriptors,
                                feat_pts,
                                camera->K());
}


VisibilityGraph Sfm::build_landmarks_graph(const std::vector<MatchAdjacent>& ma,
                                           std::vector<std::vector<Feature::Ptr>>& feat_pts) {
    // TODO: add config
    VisibilityGraph landmarks_graph;
    for (const auto& adj : ma) {
        const i32 src_idx = adj.src_idx;
        if (adj.dst_idx < 0)
            continue;
        const i32 dst_idx = adj.dst_idx;
        const auto& match = adj.match;
        const auto match_size = match.size();

        std::unordered_map<ui64, Landmark::Ptr> lms;
        for (i32 i = 0; i < match_size; ++i) {
            const auto src_pt = feat_pts[src_idx].at(match.at(i).queryIdx)->position.pt;
            const auto dst_pt = feat_pts[dst_idx].at(match.at(i).trainIdx)->position.pt;
            const auto pt_camera = project_px_point(camera, src_pt, dst_pt);
            Vec3 pt_world = Vec3::Zero();
            const bool result = triangulation(frames.at(src_idx)->pose(), frames.at(dst_idx)->pose(),
                                              pt_camera, 1.5e-1, pt_world);
            if (result && pt_world.z() > 0) {
                const auto pt_idx = match.at(i).trainIdx;
                auto feat_pt = feat_pts[dst_idx].at(pt_idx);
                feat_pt->frame = frames[dst_idx];
                Landmark::Ptr landmark = make_landmark(feat_pt, pt_world);
                lms.insert({pt_idx, landmark});
            }
        }
        landmarks_graph.insert({dst_idx, lms});
    }

    return landmarks_graph;
}


void Sfm::estimation_motion(const std::vector<MatchAdjacent>& matching,
                            std::vector<std::vector<Feature::Ptr>>& feat_pts) {
    MotionEstimationRansac::Ptr me_ransac = std::make_shared<MotionEstimationRansac>();
    
    me_ransac->estimate_motion(frames, matching, feat_pts, camera);

    this->landmarks = build_landmarks_graph(matching, feat_pts);

    MotionEstimationOptimization::Ptr me_optim = std::make_shared<MotionEstimationOptimization>();

    me_optim->estimate_motion(landmarks, frames, camera);

    for (auto& cam : landmarks) {
        for (auto lm_it = cam.second.begin(); lm_it != cam.second.end(); ) {
            const auto landmark = lm_it->second;
            if (landmark->pose().z() < 0)
                lm_it = cam.second.erase(lm_it);
            else 
                ++lm_it;
        }
    }
}


};
