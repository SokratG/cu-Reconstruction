#ifndef CUREC_LIB_OPTIMIZER_HPP
#define CUREC_LIB_OPTIMIZER_HPP


#include "landmark.hpp"
#include "keyframe.hpp"
#include "camera.hpp"
#include <memory>

namespace curec {

enum class TypeReprojectionError {
    REPROJECTION_RT = 0,
    REPROJECTION_FOCAL_RT = 1,
    UNKNOWN
};

class Optimizer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<Optimizer>;

    virtual void build_blocks(const VisibilityGraph& landmarks, 
                              const std::vector<KeyFrame::Ptr>& frames,
                              const Camera::Ptr camera) = 0;
    virtual void optimize(const i32 n_iteration, const i32 num_threads, const bool report) = 0;
    virtual void store_result(VisibilityGraph& landmarks, std::vector<KeyFrame::Ptr>& frames) = 0;
};

};

#endif