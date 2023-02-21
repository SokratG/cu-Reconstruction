#ifndef CUPHOTO_LIB_FEATURE_DETECTOR_HPP
#define CUPHOTO_LIB_FEATURE_DETECTOR_HPP

#include "config.hpp"
#include "feature.hpp"
#include "keyframe.hpp"

#include <vector>

#include <opencv2/features2d/features2d.hpp>

namespace cuphoto {


enum class FeatureDetectorBackend {
    ORB = 0,
    SIFT,
    UNKNOWN
};


class FeatureDetector {
public:
    FeatureDetector(const FeatureDetectorBackend backend,
                    const Config& cfg);

    bool detectAndCompute(const KeyFrame::Ptr frame, std::vector<Feature::Ptr>& feature_pts, cv::cuda::GpuMat& descriptor);
    
protected:
    cv::Ptr<cv::Feature2D> create_orb(const Config& cfg);
    cv::Ptr<cv::Feature2D> create_sift(const Config& cfg);
private:
    cv::Ptr<cv::Feature2D> detector;
    i32 min_keypoints;
};

};


#endif