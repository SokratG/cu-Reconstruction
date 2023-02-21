#ifndef CUPHOTO_LIB_ME_HPP
#define CUPHOTO_LIB_ME_HPP

#include "types.hpp"
#include "config.hpp"
#include "keyframe.hpp"
#include "landmark.hpp"
#include "camera.hpp"
#include "visibility_graph.hpp"
#include <memory>

namespace cuphoto {

struct MatchAdjacent;


enum class TypeMotion {
    POSE_POINT = 0,
    POSE = 1,
    POSE_ICP = 2,
    UNKNOWN
};

class MotionEstimationRansac {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<MotionEstimationRansac>;

    MotionEstimationRansac(const Config& cfg);

    bool estimate_motion(std::vector<KeyFrame::Ptr>& frames,
                         const std::vector<MatchAdjacent>& ma,
                         const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                         const Camera::Ptr camera);
private:
    void estimate_ransac(const std::vector<cv::Point2d>& src, 
                         const std::vector<cv::Point2d>& dst,
                         const cv::Mat K,
                         Mat3& R, Vec3& t);
    r64 prob;
    r64 threshold;
};

class MotionEstimationICP {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<MotionEstimationICP>;

    bool estimate_motion(std::vector<KeyFrame::Ptr>& frames,
                         const std::vector<MatchAdjacent>& ma,
                         const std::unordered_map<i32, ConnectionPoints>& pts3D);
private:
    void estimate_icp(const ConnectionPoints& pts, Mat3& R, Vec3& t);
};


class MotionEstimationOptimization {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<MotionEstimationOptimization>;

    bool estimate_motion(std::vector<Landmark::Ptr>& landmarks,
                         std::vector<KeyFrame::Ptr>& frames,
                         const VisibilityGraph& vis_graph,
                         const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                         const Camera::Ptr camera,
                         const Config& cfg);

    bool estimate_motion(std::vector<KeyFrame::Ptr>& frames,
                         const std::vector<MatchAdjacent>& ma,
                         const std::unordered_map<i32, ConnectionPoints>& pts3D,
                         const Config& cfg);
};


bool triangulation(const SE3& src_pose,
                   const SE3& dst_pose,
                   const std::pair<Vec3, Vec3>& points,
                   const r64 confidence_thrshold,
                   Vec3 &pt_world);


};


#endif