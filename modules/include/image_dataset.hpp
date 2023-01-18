#ifndef CUREC_LIB_IMAGE_DATASET_HPP
#define CUREC_LIB_IMAGE_DATASET_HPP

#include "types.hpp"
#include "utils.hpp"
#include "cr_exception.hpp"
#include <string>
#include <set>
#include <opencv4/opencv2/core.hpp>

namespace curec {

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
            throw CuRecException("The given directory is empty or not contain images!");
        current_file = files.begin();
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