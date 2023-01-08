#include "rgbd_dataset.hpp"
#include "cr_exception.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <glog/logging.h>

namespace curec {

RGBDDataset::RGBDDataset(const std::string& data_path) : ImageDataset(data_path) {
    set_data_directory(data_path);
}


void RGBDDataset::set_data_directory(const std::string& data_path) {
    this->data_path(data_path);
    const std::filesystem::path dir{data_path};
    for (const auto& entry : std::filesystem::directory_iterator(dir, 
         std::filesystem::directory_options::skip_permission_denied)) {

        const std::string ext = entry.path().extension();
        auto find_res = extensions.find(ext);
        if (find_res == extensions.end())
            continue;
        
        const std::filesystem::file_status ft(status(entry));
        const auto type = ft.type();
        if (type == std::filesystem::file_type::directory ||
            type == std::filesystem::file_type::fifo || 
            type == std::filesystem::file_type::socket ||
            type == std::filesystem::file_type::unknown) {
			continue;
        } else {
            files.insert(canonical(entry.path()).string());
        }
    }
    current_file = files.begin();
}

RGBD RGBDDataset::get_next() {
    if (current_file == files.end()) {
        LOG(WARNING) << "All files in directory was traversed! Use reset for iterate again.";
        return {cv::Mat(), cv::Mat()};
    }
    cv::Mat rgb = cv::imread(*current_file, cv::IMREAD_COLOR);
    std::advance(current_file, 1);
    cv::Mat depth = cv::imread(*current_file, cv::IMREAD_UNCHANGED);
    std::advance(current_file, 1);
    return {rgb, depth};
}

i32 RGBDDataset::num_files() const {
    return files.size();
}

void RGBDDataset::reset() {
    if (files.empty()) {
        throw CuRecException("The given directory is empty or not contain images!");
    }
    current_file = files.begin();
}

};