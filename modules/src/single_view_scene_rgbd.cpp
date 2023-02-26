#include "single_view_scene_rgbd.hpp"


namespace cuphoto {

SingleViewSceneRGBD::SingleViewSceneRGBD(const Camera::Ptr camera, const Camera::Ptr _depth_camera) :
                                         SingleViewScene(camera), depth_camera(_depth_camera)  {

}

void SingleViewSceneRGBD::reconstruct_scene(const cv::cuda::GpuMat rgb,  const cv::cuda::GpuMat depth, 
                                            const Config& cfg) {
    // TODO
}


};