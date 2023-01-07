#ifndef CUREC_LIB_IMAGE_DATASET_HPP
#define CUREC_LIB_IMAGE_DATASET_HPP

#include "types.hpp"
#include <string>
#include <set>
#include <opencv4/opencv2/core.hpp>

namespace curec {

template<typename T>
class ImageDataset {
public:
    ImageDataset(const std::string& data_path) : _data_path(data_path) {}

    virtual T get_next() = 0;
    virtual i32 num_files() const = 0;

    std::string data_path() const {
        return _data_path;
    }
protected:
    void data_path(const std::string& data_path) {
        _data_path = data_path;
    }
    std::string _data_path;
    const std::set<std::string> extensions {".jpg", ".jpeg", ".png", ".tiff"};
};


};



#endif