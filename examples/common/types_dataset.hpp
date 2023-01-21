#ifndef TYPES_DATASET_HPP
#define TYPES_DATASET_HPP

#include <opencv2/core/cuda.hpp>
#include <tuple>

namespace cuphoto {

using RGB = cv::cuda::GpuMat;
using DEPTH = cv::cuda::GpuMat;
using RGBD = std::tuple<RGB, DEPTH>;

}


#endif