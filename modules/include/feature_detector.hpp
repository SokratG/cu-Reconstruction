#ifndef CUREC_LIB_FEATURE_DETECTOR_HPP
#define CUREC_LIB_FEATURE_DETECTOR_HPP

#include "feature.hpp"
#include "keyframe.hpp"

#include <string_view>
#include <vector>

#include <opencv2/features2d/features2d.hpp>

namespace curec {


// TODO add new GPU feature detector backends
enum class FeatureDetectorBackend {
    ORB,
    SIFT,
    UNKNOWN
};


class FeatureDetector {
public:
    FeatureDetector(const FeatureDetectorBackend backend, const std::string_view config);

    bool detectAndCompute(const KeyFrame::Ptr frame, std::vector<Feature::Ptr>& feature_pts, cv::cuda::GpuMat& descriptor);
    
protected:
    cv::Ptr<cv::Feature2D> create_orb(const std::string_view config);
    cv::Ptr<cv::Feature2D> create_sift(const std::string_view config);
private:
    cv::Ptr<cv::Feature2D> detector;
    i32 min_keypoints;
};

};


#endif