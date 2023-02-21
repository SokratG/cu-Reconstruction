#include "config.hpp"

#include <glog/logging.h>


namespace cuphoto {

bool Config::set_parameter_file(const std::string& filename) {    
    file.open(filename, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
    if (!file.isOpened()) {
        LOG(ERROR) << "parameter file " << filename << " does not exist.";
        file.release();
        return false;
    }

    return true;
}

Config::~Config() {
    if (file.isOpened())
        file.release();
}

}