#ifndef CUREC_LIB_FEATURE_HPP
#define CUREC_LIB_FEATURE_HPP

#include <opencv2/core/core.hpp>

namespace curec {

class KeyFrame;

// 2D feature points
// will be associated with a map point after triangulation
struct Feature {
public:
    using Ptr = std::shared_ptr<Feature>;

    // The frame holding the feature
    std::weak_ptr<KeyFrame> frame;

    // 2D extraction position
    cv::KeyPoint position;

    // TODO: add descriptor data
    
    // Whether it is an abnormal point
    bool is_outlier = false;

public:
    Feature() {}

    Feature(std::shared_ptr<KeyFrame> _frame, const cv::KeyPoint& kp)
        : frame(_frame), position(kp) {}
};

}

#endif