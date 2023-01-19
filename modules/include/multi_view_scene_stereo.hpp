#ifndef CUREC_LIB_MUTLI_VIEW_SCENE_STEREO_HPP
#define CUREC_LIB_MUTLI_VIEW_SCENE_STEREO_HPP

#include "multi_view_scene.hpp"

namespace curec {

class MultiViewSceneStereo : public MultiViewScene {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<MultiViewSceneStereo>;

    MultiViewSceneStereo(const Camera::Ptr camera);
    bool add_frame(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right);
    void reconstruct_scene() override;
};

}


#endif