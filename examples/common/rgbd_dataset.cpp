#include "rgbd_dataset.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <glog/logging.h>

namespace cuphoto {

RGBDDataset::RGBDDataset(const std::string& data_path, const r64 _depth_scale) : 
                         ImageDataset(data_path), depth_scale(_depth_scale)
{

}


RGBD RGBDDataset::get_next() {
    if (current_file == files.end()) {
        LOG(WARNING) << "All files in directory was traversed! Use reset for iterate again.";
        return {cv::cuda::GpuMat(), cv::cuda::GpuMat()};
    }
    cv::Mat cpu_bgr = cv::imread(*current_file, cv::IMREAD_COLOR);
    cv::cuda::GpuMat gpu_bgr(cpu_bgr), gpu_rgb;

    cv::cuda::cvtColor(gpu_bgr, gpu_rgb, cv::COLOR_BGR2RGB);
    std::advance(current_file, 1);
    cv::Mat cpu_depth = cv::imread(*current_file, cv::IMREAD_ANYDEPTH); // cv::IMREAD_ANYDEPTH
    cv::cuda::GpuMat gpu_depth(cpu_depth);
    gpu_depth.convertTo(gpu_depth, CV_32F, depth_scale);
    std::advance(current_file, 1);
    return {gpu_rgb, gpu_depth};
}

i32 RGBDDataset::num_files() const {
    return files.size() / 2;
}

};