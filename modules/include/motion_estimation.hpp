#ifndef CUREC_LIB_ME_HPP
#define CUREC_LIB_ME_HPP

#include "types.hpp"
#include "keyframe.hpp"
#include "feature_matcher.hpp"
#include "landmark.hpp"
#include "camera.hpp"
#include <memory>
#include <unordered_map>

namespace curec {

class MotionEstimation {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<MotionEstimation>;
    using VisibilityGraph = std::unordered_map<ui32, std::unordered_map<ui64, Landmark::Ptr>>;

    MotionEstimation() {}

    bool estimate_motion_ransac(std::vector<KeyFrame::Ptr>& frames,
                                const std::vector<MatchAdjacent>& ma,
                                const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                const Camera::Ptr camera,
                                VisibilityGraph& landmarks);
    bool estimate_motion_non_lin_opt();
private:
    void estimate_ransac(const std::vector<cv::Point2d>& src, 
                         const std::vector<cv::Point2d>& dst,
                         const cv::Mat K,
                         Mat3& R, Vec3& t);
};

};


#endif