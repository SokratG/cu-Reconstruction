#include "depth_estimator_mono.hpp"
#include "monodepth_net.hpp"

#include "cp_exception.hpp"

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <glog/logging.h>

namespace cuphoto {


DepthEstimatorMono::DepthEstimatorMono(const Config& cfg)  {
    MonoDepthNN::NetworkType nn_type = static_cast<MonoDepthNN::NetworkType>(cfg.get<i32>("modelnetwork.type", 0));
    depth_estimator = std::make_shared<MonoDepthNN>(cfg, nn_type);
}

cv::cuda::GpuMat DepthEstimatorMono::process(const cv::cuda::GpuMat& frame, const bool equalize_hist) {
    if (frame.empty()) {
        throw CuPhotoException("Given image is empty!");
    }

    cv::cuda::GpuMat result;
    const bool res = depth_estimator->process(frame, result);

    if (equalize_hist) {
        cv::cuda::GpuMat equalize_result;
        result.convertTo(equalize_result, CV_8UC1);
        cv::cuda::equalizeHist(equalize_result, result);
    }
    cv::cuda::resize(result, result, cv::Size(frame.cols, frame.rows), cv::INTER_CUBIC);    

    return result;
}


};