#include "multi_view_scene.hpp"

namespace curec {

MultiViewSceneStereo::MultiViewSceneStereo(const Camera::Ptr _camera) : MultiViewScene(_camera) {

}

bool MultiViewSceneStereo::add_frame(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right) {
    return true;
}

void MultiViewSceneStereo::reconstruct_scene() {

}

};