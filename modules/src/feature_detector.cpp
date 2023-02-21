#include "feature_detector.hpp"
#include "cp_exception.hpp"
#include "cuda_sift.hpp"

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include <glog/logging.h>

namespace cuphoto {

FeatureDetector::FeatureDetector(const FeatureDetectorBackend backend,
                                 const Config& cfg) : 
                                 min_keypoints(cfg.get<i32>("feature.threshold_min_keypoints", 200))
{
    switch (backend) {
        case FeatureDetectorBackend::ORB:
            detector = create_orb(cfg);
            break;
        case FeatureDetectorBackend::SIFT:
            detector = create_sift(cfg);
            break;
        default:
            throw CuPhotoException("The given feature detector backend is not allowed!");
    }
}

cv::Ptr<cv::Feature2D> FeatureDetector::create_orb(const Config& cfg) {
    // TODO: add config read ORB parameters
    return cv::cuda::ORB::create(
        cfg.get<i32>("feature.orb.num_points", 1000),
        cfg.get<r32>("feature.orb.scale_pyr", 1.2),
        cfg.get<i32>("feature.orb.num_pyr", 8),
        cfg.get<i32>("feature.orb.edge_threshold", 31),
        cfg.get<i32>("feature.orb.start_level", 0),
        cfg.get<i32>("feature.orb.wta_k", 3),
        cv::ORB::HARRIS_SCORE,
        cfg.get<i32>("feature.orb.patch_size", 31),
        cfg.get<i32>("feature.orb.fast_threshold", 20)
    );
}

cv::Ptr<cv::Feature2D> FeatureDetector::create_sift(const Config& cfg) {
    SiftParams sp;
    sp.maxKeypoints = cfg.get<i32>("feature.sift.max_keypoints", 1500);
    sp.numOctaves = cfg.get<i32>("feature.sift.num_octaves", 6);
    sp.initBlur = cfg.get<r32>("feature.sift.init_blur", 1.0);
    sp.thresh = cfg.get<r32>("feature.sift.thresh", 1.7);
    sp.minScale = cfg.get<r32>("feature.sift.min_scale", 0.0);
    sp.upScale = static_cast<bool>(cfg.get<i32>("feature.sift.up_scale", 0));

    return CudaSift::create(sp);
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