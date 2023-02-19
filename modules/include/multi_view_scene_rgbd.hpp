#ifndef CUPHOTO_LIB_MUTLI_VIEW_SCENE_RGBD_HPP
#define CUPHOTO_LIB_MUTLI_VIEW_SCENE_RGBD_HPP

#include "multi_view_scene.hpp"

#include <tuple>
#include <string>

namespace cuphoto {


class MultiViewSceneRGBD : public MultiViewScene {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<MultiViewSceneRGBD>;
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

    MultiViewSceneRGBD(const Camera::Ptr camera, const Camera::Ptr depth_camera);
    bool add_frame(const cv::cuda::GpuMat rgb, const cv::cuda::GpuMat depth);
    void reconstruct_scene() override;

protected:
    std::tuple<std::vector<RGB>, std::vector<cv::Mat>> split_rgbd();
    void filter_outlier_frames(std::vector<std::vector<Feature::Ptr>>& feat_pts,
                               std::vector<cv::cuda::GpuMat>& descriptors);
    virtual void estimate_motion();
    void build_point_cloud();

private:
    std::vector<RGBD::Ptr> rgbd_frames;
    Camera::Ptr depth_camera;
};

}


#endif