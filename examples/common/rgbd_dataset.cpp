#include "rgbd_dataset.hpp"
#include "cr_exception.hpp"
#include "utils.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <glog/logging.h>

namespace curec {

RGBDDataset::RGBDDataset(const std::string& data_path) : ImageDataset(data_path) {

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
    cv::Mat cpu_depth = cv::imread(*current_file, cv::IMREAD_UNCHANGED); // cv::IMREAD_ANYDEPTH
    cv::cuda::GpuMat gpu_depth(cpu_depth);
    std::advance(current_file, 1);
    return {gpu_rgb, gpu_depth};
}

i32 RGBDDataset::num_files() const {
    return files.size() / 2;
}

};