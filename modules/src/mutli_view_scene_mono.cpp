#include "multi_view_scene.hpp"

namespace curec {

MultiViewSceneMono::MultiViewSceneMono(const Camera::Ptr _camera) : MultiViewScene(_camera) {

}

bool MultiViewSceneMono::add_frame(const cv::cuda::GpuMat rgb) {
    return true;
}

void MultiViewSceneMono::reconstruct_scene() {

}

};