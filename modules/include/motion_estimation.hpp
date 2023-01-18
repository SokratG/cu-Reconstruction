#ifndef CUREC_LIB_ME_HPP
#define CUREC_LIB_ME_HPP

#include "types.hpp"
#include "keyframe.hpp"
#include "landmark.hpp"
#include "camera.hpp"
#include "visibility_graph.hpp"
#include <memory>

namespace curec {

struct MatchAdjacent;

class MotionEstimationRansac {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<MotionEstimationRansac>;

    bool estimate_motion(std::vector<KeyFrame::Ptr>& frames,
                         const std::vector<MatchAdjacent>& ma,
                         const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                         const Camera::Ptr camera);
private:
    void estimate_ransac(const std::vector<cv::Point2d>& src, 
                         const std::vector<cv::Point2d>& dst,
                         const cv::Mat K,
                         Mat3& R, Vec3& t);
};


class MotionEstimationOptimization {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<MotionEstimationOptimization>;

    bool estimate_motion(std::vector<Landmark::Ptr>& landmarks,
                         std::vector<KeyFrame::Ptr>& frames,
                         const VisibilityGraph& vis_graph,
                         const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                         const Camera::Ptr camera);
};


};


#endif