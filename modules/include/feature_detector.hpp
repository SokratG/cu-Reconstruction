#ifndef CUREC_LIB_FEATURE_DETECTOR_HPP
#define CUREC_LIB_FEATURE_DETECTOR_HPP

#include "feature.hpp"
#include "keyframe.hpp"
#include <string>
#include <vector>
#include <opencv4/opencv2/features2d.hpp>

namespace curec {


// TODO add new GPU feature detector backends
enum class FeatureDetectorBackend {
    ORB,
    SIFT,
    UNKNOWN
};


class FeatureDetector {
public:
    FeatureDetector(const FeatureDetectorBackend backend, const std::string& config);

    bool detect(const KeyFrame::Ptr frame, std::vector<Feature>& feature_pts, cv::Mat& descriptor);
    
protected:
    cv::Ptr<cv::Feature2D> create_orb(const std::string& config);
    cv::Ptr<cv::Feature2D> create_sift(const std::string& config);
private:
    cv::Ptr<cv::Feature2D> detector;
    i32 min_keypoints;
};

};


#endif