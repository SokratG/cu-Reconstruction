#ifndef CUPHOTO_LIB_SINGLE_VIEW_SCENE_RGBD_HPP
#define CUPHOTO_LIB_SINGLE_VIEW_SCENE_RGBD_HPP

#include "single_view_scene.hpp"

#include "config.hpp"

namespace cuphoto {

class SingleViewSceneRGBD : public SingleViewScene {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<SingleViewSceneRGBD>;
protected:
    using RGB = KeyFrame::Ptr;
    using DEPTH = KeyFrame::Ptr;
    struct RGBD {
        using Ptr = std::shared_ptr<RGBD>;
        RGB rgb;
        DEPTH depth;
        RGBD(RGB _rgb, DEPTH _depth) : rgb(_rgb), depth(_depth) {}
    };
public:
    SingleViewSceneRGBD(const Camera::Ptr camera, const Camera::Ptr depth_camera);

    void reconstruct_scene(const cv::cuda::GpuMat rgb,  const cv::cuda::GpuMat depth, const Config& cfg);

private:
    RGBD::Ptr rgbd_frame;
    Camera::Ptr depth_camera;
};

};

#endif