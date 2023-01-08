#include "sfm.hpp"
#include "feature_detector.hpp"
#include "feature_matcher.hpp"
#include "visibility_graph.hpp"
 #include <opencv2/core/eigen.hpp>
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
    frames.push_back(kframe_ptr);
    return true;
}

void Sfm::build_landmark_graph() {
    FeatureDetector fd(FeatureDetectorBackend::SIFT, std::string());
    std::vector<cv::Mat> descriptors;
    std::vector<std::vector<Feature>> feat_pts;
    for (auto frame : frames) {
        std::vector<Feature> kpts;
        cv::Mat descriptor;
        fd.detect(frame, kpts, descriptor);
        descriptors.emplace_back(descriptor);
        feat_pts.emplace_back(kpts);
    }

    std::vector<MatchAdjacent> matching = feature_matching(FeatureMatcherBackend::BRUTEFORCE,
                                                           descriptors);
    
    LOG(INFO) << "matching size: " << matching.size();
    cv::Mat K;
    cv::eigen2cv(camera->K(), K);
    matching = ransac_filter_outlier(matching, feat_pts, K);

    LOG(INFO) << "matching size after filter outliers: " << matching.size();

    VisibilityGraph v_graph;
    v_graph.build_nodes(matching, feat_pts);

}

};
