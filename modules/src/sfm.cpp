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
    const Vec3f color = feat_pt->frame.lock()->get_color(pt2);
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

        for (auto i = 0; i < match_size; ++i) {
            const auto pt_src_idx = match.at(i).queryIdx; 
            const auto pt_dst_idx = match.at(i).trainIdx;

            const auto src_pt = feat_pts[src_idx].at(pt_src_idx)->position.pt;
            const auto dst_pt = feat_pts[dst_idx].at(pt_dst_idx)->position.pt;

            const auto key1 = gen_combined_key(src_idx, pt_src_idx); 
            const auto key2 = gen_combined_key(dst_idx, pt_dst_idx); 

            auto visitView1 = landmarks_graph.find(key1);
            auto visitView2 = landmarks_graph.find(key2);

            if (visitView1 != landmarks_graph.end() && visitView2 != landmarks_graph.end()) {
                
                // remove previous view points
                if (visitView1->second->landmark_idx() != visitView2->second->landmark_idx()) {
                    landmarks_graph.erase(visitView1);
                    landmarks_graph.erase(visitView2);
                }
                continue;
            }
            
            auto feat_pt = feat_pts[dst_idx].at(pt_dst_idx);
            feat_pt->frame = frames[dst_idx];
            Vec3 pt_world = Vec3::Zero();
            const auto pt_camera = project_px_point(camera, src_pt, dst_pt);
            const bool result = triangulation(frames.at(src_idx)->pose(), frames.at(dst_idx)->pose(),
                                              pt_camera, 1.5e-2, pt_world);
            const bool is_valid = result && !feat_pt->outlier();
            ui32 visibility_idx = 0;
            if (visitView1 != landmarks_graph.end())
                visibility_idx = visitView1->second->landmark_idx();
            else if (visitView2 != landmarks_graph.end())
                visibility_idx = visitView2->second->landmark_idx();
            else {
                visibility_idx = landmarks.size();
                if (is_valid) {
                    Landmark::Ptr landmark = make_landmark(feat_pt, pt_world);
                    landmarks.emplace_back(landmark);
                }
            }
            
            if (is_valid) {
                if (visitView1 == landmarks_graph.end())
                    landmarks_graph[key1] = std::make_shared<VisibilityNode>(src_idx, pt_src_idx, visibility_idx);
                if (visitView2 == landmarks_graph.end())
                    landmarks_graph[key2] = std::make_shared<VisibilityNode>(dst_idx, pt_dst_idx, visibility_idx);
            }
        }    
    }

    return landmarks_graph;
}


void Sfm::estimation_motion(const std::vector<MatchAdjacent>& matching,
                            std::vector<std::vector<Feature::Ptr>>& feat_pts) {
    MotionEstimationRansac::Ptr me_ransac = std::make_shared<MotionEstimationRansac>();
    
    me_ransac->estimate_motion(frames, matching, feat_pts, camera);

    vis_graph = build_landmarks_graph(matching, feat_pts);

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
