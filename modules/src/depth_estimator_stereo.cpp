#include "depth_estimator_stereo.hpp"
#include "CudaUtils/cudaUtility.cuh"
#include "cuda/util.cuh"

#include <opencv2/cudastereo.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <opencv2/cudaarithm.hpp>

#include <glog/logging.h>

namespace cuphoto {

DepthEstimatorStereo::DepthEstimatorStereo(const Camera::Ptr _camera_left, 
                                           const Camera::Ptr _camera_right) :
                                           camera_left(_camera_left), camera_right(_camera_right) 
{
    
}



// ============================================== SEMI GLOBAL MATCHING =============================================
static i32 sgm_mode(const i32 mode) {
    switch(mode) {
        case 0: return cv::cuda::StereoSGM::MODE_SGBM;
        case 1: return cv::cuda::StereoSGM::MODE_HH;
        case 2: return cv::cuda::StereoSGM::MODE_SGBM_3WAY;
        case 3: return cv::cuda::StereoSGM::MODE_HH4;
        default: {
            LOG(WARNING) << "The given mode in stereo global matching is available. The default was MODE_HH4.";
            return cv::cuda::StereoSGM::MODE_HH4;
        }   
    }
}

DepthEstimatorStereoSGM::DepthEstimatorStereoSGM(const Camera::Ptr camera_left, const Camera::Ptr camera_right, 
                                                   const Config& cfg) : DepthEstimatorStereo(camera_left, camera_right) 
{
    param.min_disparity = cfg.get<i32>("stereo.cuda.sgm.min_disparity", 0);
    param.num_disparities = cfg.get<i32>("stereo.cuda.sgm.num_disparities", 128);
    param.regularization_smoothness_p1 = cfg.get<i32>("stereo.cuda.sgm.regularization_smoothness_p1", 10);
    param.regularization_smoothness_p2 = cfg.get<i32>("stereo.cuda.sgm.regularization_smoothness_p2", 120);
    param.uniqueness_ratio = cfg.get<i32>("stereo.cuda.sgm.uniqueness_ratio", 5);
    param.mode = sgm_mode(cfg.get<i32>("stereo.cuda.sgm.mode", 3));

    depth_scale = cfg.get<r32>("depth_scale", 1.0);
}

cv::cuda::GpuMat DepthEstimatorStereoSGM::estimate_depth(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right,
                                                          const r32 baseline) {
    const auto disp_map = estimate_disparity(left, right);

    if (disp_map.empty()) 
        return cv::cuda::GpuMat();
    
    const size_t alloc_size = disp_map.rows * disp_map.cols * sizeof(r32);
    r32* output_depth_data = nullptr;
    if(CUDA_FAILED(cudaMalloc(&output_depth_data, alloc_size))) {
        LOG(ERROR) << "Can't allocate GPU memory size: " << alloc_size;
        throw std::bad_alloc();
    }

    const r32 focal = camera_left->fx();

    if(CUDA_FAILED(disparity_to_depth(disp_map, output_depth_data, focal, baseline, depth_scale))) {
        LOG(WARNING) << "Can't convert disparity map to depth!";
        return cv::cuda::GpuMat();
    }

    cv::cuda::GpuMat depth_map = cv::cuda::GpuMat(left.size(), CV_32FC1, output_depth_data);

    return depth_map;
}

cv::cuda::GpuMat DepthEstimatorStereoSGM::estimate_disparity(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right) {
    if ((left.type() != CV_8UC3 && left.type() != CV_8UC1) || (right.type() != CV_8UC3 && right.type() != CV_8UC1)) {
        LOG(ERROR) << "The given image type is not allowed. Use CV_8UC3 or CV_8UC1";
        return cv::cuda::GpuMat();
    }
    
    cv::Ptr<cv::cuda::StereoSGM> sgm = cv::cuda::createStereoSGM(param.min_disparity,
                                                                 param.num_disparities,
                                                                 param.regularization_smoothness_p1,
                                                                 param.regularization_smoothness_p2,
                                                                 param.uniqueness_ratio,
                                                                 param.mode);
    
    cv::cuda::GpuMat left_gray, right_gray;
    if (left.type() == CV_8UC1)
        left_gray = left;
    else
        cv::cuda::cvtColor(left, left_gray, cv::COLOR_RGB2GRAY);
    
    if (right.type() == CV_8UC1)
        right_gray = right;
    else
        cv::cuda::cvtColor(right, right_gray, cv::COLOR_RGB2GRAY);
    
    cv::cuda::GpuMat disparity_map;
    sgm->compute(left_gray, right_gray, disparity_map);

    return disparity_map;
}


// ============================================== PATCH MATCHING =============================================
DepthEstimatorStereoPM::DepthEstimatorStereoPM(const Camera::Ptr camera_left, const Camera::Ptr camera_right, 
                                               const Config& cfg) : DepthEstimatorStereo(camera_left, camera_right)
{
    // TODO
}


cv::cuda::GpuMat DepthEstimatorStereoPM::estimate_depth(const cv::cuda::GpuMat left, const cv::cuda::GpuMat right,
                                                        const r32 baseline) {
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