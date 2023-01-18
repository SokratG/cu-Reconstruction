#ifndef CUREC_LIB_MUTLI_VIEW_SCENE_HPP
#define CUREC_LIB_MUTLI_VIEW_SCENE_HPP

#include "camera.hpp"
#include "keyframe.hpp"
#include <opencv2/core/cuda.hpp>

namespace curec {


class MultiViewScene {
public:
    MultiViewScene(const Camera::Ptr camera);

    virtual void reconstruct_scene() = 0;
    
    // MOTION ESTIMATION
    // OPTIONAL: ESTIMATE DEPTH
    // POINT CLOUD FROM DEPTH
    // SURFACE FROM CLOUD
    // TEXTURE MAPPING
protected:
    Camera::Ptr camera;
};


class MultiViewSceneRGBD : public MultiViewScene {
public:
    MultiViewSceneRGBD(const Camera::Ptr camera);
    bool add_frame(const cv::cuda::GpuMat rgb, const cv::cuda::GpuMat depth);
    void reconstruct_scene() override;
private:
    std::vector<KeyFrame::Ptr> rgb_frames;
    std::vector<KeyFrame::Ptr> depth_frames;
};


class MultiViewSceneMono : public MultiViewScene {
public:
    MultiViewSceneMono(const Camera::Ptr camera);
    bool add_frame(const cv::cuda::GpuMat rgb);
    void reconstruct_scene() override;

};



class MultiViewSceneStereo : public MultiViewScene {
public:
    MultiViewSceneStereo(const Camera::Ptr camera);
    bool add_frame(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right);
    void reconstruct_scene() override;
};


};

#endif