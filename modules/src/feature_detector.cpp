#include "feature_detector.hpp"
#include "cp_exception.hpp"
#include "cuda_sift.hpp"

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include <glog/logging.h>

namespace cuphoto {

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
            throw CuPhotoException("The given feature detector backend is not allowed!");
    }
}

cv::Ptr<cv::Feature2D> FeatureDetector::create_orb(const std::string_view config) {
    // TODO: add config read ORB parameters
    return cv::cuda::ORB::create(1000, 1.2, 8, 31, 0, 3, cv::ORB::HARRIS_SCORE);
}

cv::Ptr<cv::Feature2D> FeatureDetector::create_sift(const std::string_view config) {
    // TODO: add config read SIFT parameters
    return CudaSift::create();
}


bool FeatureDetector::detectAndCompute(const KeyFrame::Ptr frame, 
                                       std::vector<Feature::Ptr>& feature_pts, 
                                       cv::cuda::GpuMat& descriptor) {
    const cv::cuda::GpuMat image = frame->frame();
    if (image.empty()) 
        throw CuPhotoException("The given image has no data!");
    std::vector<cv::KeyPoint> k_pts;
    cv::cuda::GpuMat gray;
    cv::cuda::cvtColor(image, gray, cv::COLOR_RGB2GRAY);

    detector->detectAndCompute(gray, cv::noArray(), k_pts, descriptor);

    if (k_pts.size() < min_keypoints) {
        LOG(WARNING) << "detected number of key points: " << k_pts.size() <<  ", less then minimum required: " << min_keypoints;
        return false;
    }
    
    feature_pts = std::vector<Feature::Ptr>();
    for (auto idx = 0; idx < k_pts.size(); ++idx) {
        feature_pts.emplace_back(std::make_shared<Feature>(frame, k_pts[idx]));
    }

    if (descriptor.depth() != CV_32F)
        descriptor.convertTo(descriptor, CV_32F);

    return true;
}

};