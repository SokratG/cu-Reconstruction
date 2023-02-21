#ifndef CUPHOTO_LIB_OPTIMIZER_HPP
#define CUPHOTO_LIB_OPTIMIZER_HPP

#include "config.hpp"
#include "landmark.hpp"
#include "keyframe.hpp"
#include "camera.hpp"
#include "visibility_graph.hpp"
#include <memory>

namespace cuphoto {

enum class TypeReprojectionError {
    REPROJECTION_POSE_POINT = 0,
    REPROJECTION_POSE = 1,
    UNKNOWN
};

class Optimizer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<Optimizer>;

    virtual void build_blocks_reprojection(const VisibilityGraph& vis_graph,
                                           const std::vector<Landmark::Ptr>& landmarks,
                                           const std::vector<KeyFrame::Ptr>& frames,
                                           const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                           const Camera::Ptr camera) = 0;

    virtual void build_blocks_icp(const std::unordered_map<i32, ConnectionPoints>& pts3D,
                                  const std::vector<MatchAdjacent>& ma,
                                  const std::vector<KeyFrame::Ptr>& frames) = 0;

    virtual void optimize(const i32 n_iteration, const i32 num_threads, const bool report) = 0;
    
    virtual void store_result(std::vector<Landmark::Ptr>& landmarks,
                              std::vector<KeyFrame::Ptr>& frames) = 0;
    virtual void store_result(std::vector<KeyFrame::Ptr>& frames) = 0;
};

};

#endif