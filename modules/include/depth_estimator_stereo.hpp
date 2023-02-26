#ifndef CUPHOTO_LIB_DEPTH_ESTIMATOR_STEREO_HPP
#define CUPHOTO_LIB_DEPTH_ESTIMATOR_STEREO_HPP

#include "types.hpp"
#include "config.hpp"

#include <opencv2/core/cuda.hpp>

#include <memory>

namespace cuphoto {

class DepthEstimatorStereo {
    /**
        * TODO estimate motion translation if not provided baseline !!!
    */ 
public:
    virtual cv::cuda::GpuMat estimate_depth(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right, 
                                            const r32 focal, const r32 baseline) = 0;

    virtual cv::cuda::GpuMat estimate_disparity(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right) = 0;

};


// Semi-Global Block Matching
class DepthEstimatorStereoSGBM : public DepthEstimatorStereo {
public:
    DepthEstimatorStereoSGBM(const Config& cfg);

    cv::cuda::GpuMat estimate_depth(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right,
                                    const r32 focal, const r32 baseline) override;

    cv::cuda::GpuMat estimate_disparity(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right) override;
private:
};


// Patch Matching
class DepthEstimatorStereoPM : public DepthEstimatorStereo {
public:
    DepthEstimatorStereoPM(const Config& cfg);

    cv::cuda::GpuMat estimate_depth(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right, 
                                    const r32 focal, const r32 baseline) override;

    cv::cuda::GpuMat estimate_disparity(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right) override;
private:
};



};


#endif