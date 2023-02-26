#ifndef CUPHOTO_LIB_SINGLE_VIEW_SCENE_HPP
#define CUPHOTO_LIB_SINGLE_VIEW_SCENE_HPP

#include "cuda/cuda_point_cloud.cuh"

#include "camera.hpp"
#include "keyframe.hpp"
#include "feature.hpp"
#include <opencv2/core/cuda.hpp>

namespace cuphoto {

class SingleViewScene {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<SingleViewScene>;

public:
    SingleViewScene(const Camera::Ptr camera);
    cudaPointCloud::Ptr get_point_cloud() const;
    bool store_to_ply(const std::string& ply_filepath) const;

protected:
    Camera::Ptr camera;
    cudaPointCloud::Ptr cuda_pc;
};

};


#endif