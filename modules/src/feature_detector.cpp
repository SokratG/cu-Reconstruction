#include "feature_detector.hpp"
#include "cr_exception.hpp"
#include "cuda_sift.hpp"
#include <glog/logging.h>

namespace curec {

FeatureDetector::FeatureDetector(const FeatureDetectorBackend backend, const std::string_view config) : 
                                 min_keypoints(200)
{
    switch (backend) {
        case FeatureDetectorBackend::ORB:
            detector = create_orb(config);
            break;
        case FeatureDetectorBackend::SIFT:
            detector = create_sift(config);
            break;
        default:
            throw CuRecException("The given feature detector backend is not allowed!");
    }
}

cv::Ptr<cv::Feature2D> FeatureDetector::create_orb(const std::string_view config) {
    // TODO: add config read ORB parameters
    return cv::ORB::create(1000, 1.2, 8, 31, 0, 3, cv::ORB::HARRIS_SCORE);
}

cv::Ptr<cv::Feature2D> FeatureDetector::create_sift(const std::string_view config) {
    // TODO: add config read SIFT parameters
    return CudaSift::create();
}


bool FeatureDetector::detectAndCompute(const KeyFrame::Ptr frame, std::vector<Feature::Ptr>& feature_pts, cv::Mat& descriptor) {
    cv::Mat image = frame->frame();
    if (image.empty()) 
        throw CuRecException("The given image has no data!");
    std::vector<cv::KeyPoint> k_pts;
    detector->detectAndCompute(image, cv::noArray(), k_pts, descriptor);


    if (k_pts.size() < min_keypoints) {
        LOG(WARNING) << "detected number of key points: " << k_pts.size() <<  ", less then minimum required: " << min_keypoints;
        return false;
    }
    
    feature_pts = std::vector<Feature::Ptr>();
    for (auto idx = 0; idx < k_pts.size(); ++idx) {
        feature_pts.emplace_back(std::make_shared<Feature>(frame, k_pts[idx]));
    }

    descriptor.convertTo(descriptor, CV_32F);

    return true;
}

};