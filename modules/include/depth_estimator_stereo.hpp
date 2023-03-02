#ifndef CUPHOTO_LIB_DEPTH_ESTIMATOR_STEREO_HPP
#define CUPHOTO_LIB_DEPTH_ESTIMATOR_STEREO_HPP

#include "types.hpp"
#include "config.hpp"
#include "camera.hpp"

#include <opencv2/core/cuda.hpp>

#include <memory>

namespace cuphoto {

class DepthEstimatorStereo {
public:
    DepthEstimatorStereo(const Camera::Ptr camera_left, const Camera::Ptr camera_right);

    virtual cv::cuda::GpuMat estimate_depth(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right, 
                                            const r32 baseline) = 0;

    virtual cv::cuda::GpuMat estimate_disparity(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right) = 0;
protected:
    Camera::Ptr camera_left, camera_right;
    r32 depth_scale;
};


// Semi-Global Matching
class DepthEstimatorStereoSGM : public DepthEstimatorStereo {
private:
    struct SGMParam {
        i32 min_disparity;
        i32 num_disparities;
        i32 regularization_smoothness_p1;
        i32 regularization_smoothness_p2;
        i32 uniqueness_ratio;
        i32 mode;
    };
public:
    DepthEstimatorStereoSGM(const Camera::Ptr camera_left, const Camera::Ptr camera_right, const Config& cfg);

    cv::cuda::GpuMat estimate_depth(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right,
                                    const r32 baseline) override;

    cv::cuda::GpuMat estimate_disparity(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right) override;
private:
    SGMParam param;
};


// Patch Matching
class DepthEstimatorStereoPM : public DepthEstimatorStereo {
public:
    DepthEstimatorStereoPM(const Camera::Ptr camera_left, const Camera::Ptr camera_right, const Config& cfg);

    cv::cuda::GpuMat estimate_depth(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right, 
                                    const r32 baseline) override;

    cv::cuda::GpuMat estimate_disparity(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right) override;
private:
};



};


#endif