#ifndef CUPHOTO_LIB_MUTLI_VIEW_SCENE_HPP
#define CUPHOTO_LIB_MUTLI_VIEW_SCENE_HPP

#include "cuda/cuda_point_cloud.cuh"

#include "config.hpp"
#include "camera.hpp"
#include "keyframe.hpp"
#include "feature.hpp"
#include "feature_matcher.hpp"
#include <opencv2/core/cuda.hpp>


namespace cuphoto {


class MultiViewScene {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<MultiViewScene>;
public:
    MultiViewScene(const Camera::Ptr camera);

    virtual void reconstruct_scene(const Config& cfg) = 0;
    cudaPointCloud::Ptr get_point_cloud() const;
    bool store_to_ply(const std::string& ply_filepath) const;

protected:
    void detect_feature(const std::vector<KeyFrame::Ptr>& frames,
                        const Config& cfg,
                        std::vector<std::vector<Feature::Ptr>>& feat_pts,
                        std::vector<cv::cuda::GpuMat>& descriptors);

    void matching_feature(const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                          const std::vector<cv::cuda::GpuMat>& descriptors,
                          const Config& cfg,
                          std::vector<MatchAdjacent>& matching);
protected:
    Camera::Ptr camera;
    cudaPointCloud::Ptr cuda_pc;
};

};

#endif