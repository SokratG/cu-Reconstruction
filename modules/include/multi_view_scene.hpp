#ifndef CUREC_LIB_MUTLI_VIEW_SCENE_HPP
#define CUREC_LIB_MUTLI_VIEW_SCENE_HPP

#include "camera.hpp"
#include "keyframe.hpp"
#include "feature_detector.hpp"
#include "feature_matcher.hpp"
#include <opencv2/core/cuda.hpp>

namespace curec {


class MultiViewScene {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<MultiViewScene>;
public:
    MultiViewScene(const Camera::Ptr _camera);

    virtual void reconstruct_scene() = 0;

protected:
    void detect_feature(const std::vector<KeyFrame::Ptr>& frames,
                        std::vector<std::vector<Feature::Ptr>>& feat_pts, 
                        std::vector<MatchAdjacent>& matching);
    // MOTION ESTIMATION
    // OPTIONAL: ESTIMATE DEPTH
    // POINT CLOUD FROM DEPTH
    // SURFACE FROM CLOUD
    // TEXTURE MAPPING
protected:
    Camera::Ptr camera;
};

};

#endif