#include "feature_detector.hpp"
#include "cr_exception.hpp"
#include <glog/logging.h>

namespace curec {

FeatureDetector::FeatureDetector(const FeatureDetectorBackend backend, const std::string& config) : 
                                 min_keypoints(200)
{
    switch (backend) {
        case FeatureDetectorBackend::ORB:
            detector = create_orb(config);
            break;
        default:
            throw CuRecException("The given feature detector backend is not allowed!");
    }
}

cv::Ptr<cv::Feature2D> FeatureDetector::create_orb(const std::string& config) {
    // TODO: add config read ORB parameters
    return cv::ORB::create(600, 1.2, 8, 31, 0, 3, cv::ORB::HARRIS_SCORE);
}


bool FeatureDetector::detect(const KeyFrame::Ptr frame, std::vector<Feature>& feature_pts, cv::Mat& descriptor) {
    cv::Mat image = frame->frame();
    if (image.empty()) 
        throw CuRecException("The given image has no data!");
    std::vector<cv::KeyPoint> k_pts;
    detector->detectAndCompute(image, cv::noArray(), k_pts, descriptor);

    if (k_pts.size() < min_keypoints) {
        LOG(WARNING) << "detected number of key points: " << k_pts.size() <<  ", less then minimum required: " << min_keypoints;
        return false;
    }

    feature_pts = std::vector<Feature>(k_pts.size());
    for (auto idx = 0; idx < k_pts.size(); ++idx) {
        feature_pts[idx].position = k_pts[idx];
        feature_pts[idx].frame = frame;
    }

    return true;
}

};