#ifndef CUPHOTO_LIB_IMAGE_DATASET_HPP
#define CUPHOTO_LIB_IMAGE_DATASET_HPP

#include "types.hpp"
#include "utils.hpp"
#include "cp_exception.hpp"
#include <string>
#include <set>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

namespace cuphoto {

template<typename T>
class ImageDataset {
public:
    ImageDataset(const std::string& data_path) : _data_path(data_path) {
        set_data_directory(data_path);
    }

    virtual T get_next() = 0;
    virtual i32 num_files() const {
        return files.size();
    };

    std::string data_path() const {
        return _data_path;
    }

    void set_data_directory(const std::string& data_path) {
        this->data_path(data_path);
        files = files_directory(data_path, extensions);
        current_file = files.begin();
    }

    virtual void reset() {
        if (files.empty())
            throw CuPhotoException("The given directory is empty or not contain images!");
        current_file = files.begin();
    }

    void store_image(const std::string& filepath, const cv::cuda::GpuMat& image) {
        cv::Mat cpu_image;
        image.download(cpu_image);
        const bool res = save_image(filepath, cpu_image);
        if (!res) {
            throw CuPhotoException("The given filepath or image is empty!");
        }
    }
    void store_image(const std::string& filepath, const cv::Mat& image) {
        const bool res = save_image(filepath, image);
        if (!res) {
            throw CuPhotoException("The given filepath or image is empty!");
        }
    }

    void step_on(const i32 adv_step) {
        if (adv_step >= files.size())
            return;
        std::advance(current_file, adv_step);
    }
protected:
    void data_path(const std::string& data_path) {
        _data_path = data_path;
    }
    std::string _data_path;
    const std::set<std::string> extensions {".jpg", ".jpeg", ".png", ".tiff"};
    std::set<std::string> files;
    std::set<std::string>::iterator current_file;
};


};



#endif