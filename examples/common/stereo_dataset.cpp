#include "stereo_dataset.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <glog/logging.h>


namespace cuphoto {

StereoDataset::StereoDataset(const std::string& data_path) : ImageDataset(data_path)
{

}


STEREO StereoDataset::get_next() {
    if (current_file == files.end()) {
        LOG(WARNING) << "All files in directory was traversed! Use reset for iterate again.";
        return {cv::cuda::GpuMat(), cv::cuda::GpuMat()};
    }

    cv::Mat left_bgr = cv::imread(*current_file, cv::IMREAD_COLOR);
    std::advance(current_file, 1);
    cv::Mat right_bgr = cv::imread(*current_file, cv::IMREAD_COLOR);

    cv::cuda::GpuMat gpu_left_bgr(left_bgr), gpu_right_bgr(right_bgr);
    cv::cuda::GpuMat gpu_left_rgb, gpu_right_rgb;
    cv::cuda::cvtColor(gpu_left_bgr, gpu_left_rgb, cv::COLOR_BGR2RGB);
    cv::cuda::cvtColor(gpu_right_bgr, gpu_right_rgb, cv::COLOR_BGR2RGB);

    std::advance(current_file, 1);

    return {gpu_left_rgb, gpu_right_rgb};
}


i32 StereoDataset::num_files() const {
    return files.size() / 2;
}


}