#ifndef CUPHOTO_LIB_DEPTH_ESTIMATOR_MONO_HPP
#define CUPHOTO_LIB_DEPTH_ESTIMATOR_MONO_HPP

#include "types.hpp"
#include "config.hpp"

#include <opencv2/core/cuda.hpp>

#include <memory>

namespace cuphoto {


class MonoDepthNN;


class DepthEstimatorMono {
public:
    DepthEstimatorMono(const Config& cfg);
    cv::cuda::GpuMat process(const cv::cuda::GpuMat& frame, const bool equalize_hist = false);
private:
    std::shared_ptr<MonoDepthNN> depth_estimator;
    r64 scale;
};

};

#endif