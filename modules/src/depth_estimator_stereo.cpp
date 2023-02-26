#include "depth_estimator_stereo.hpp"


namespace cuphoto {

// ============================================== SEMI GLOBAL BLOCK MATCHING =============================================
DepthEstimatorStereoSGBM::DepthEstimatorStereoSGBM(const Config& cfg) {
    // TODO
}

cv::cuda::GpuMat DepthEstimatorStereoSGBM::estimate_depth(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right,
                                                          const r32 focal, const r32 baseline) {
    cv::cuda::GpuMat depth_map;
    
    // TODO

    return depth_map;
}

cv::cuda::GpuMat DepthEstimatorStereoSGBM::estimate_disparity(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right) {
    cv::cuda::GpuMat disparity_map;
    
    // TODO

    return disparity_map;
}


// ============================================== PATCH MATCHING =============================================
DepthEstimatorStereoPM::DepthEstimatorStereoPM(const Config& cfg) {
    // TODO
}


cv::cuda::GpuMat DepthEstimatorStereoPM::estimate_depth(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right,
                                                        const r32 focal, const r32 baseline) {
    cv::cuda::GpuMat depth_map;

    // TODO

    return depth_map;
}

cv::cuda::GpuMat DepthEstimatorStereoPM::estimate_disparity(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right) {
    cv::cuda::GpuMat disparity_map;
    
    // TODO

    return disparity_map;
}


};