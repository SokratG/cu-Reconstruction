#ifndef CUPHOTO_LIB_CONFIG_HPP
#define CUPHOTO_LIB_CONFIG_HPP

#include <opencv2/core/persistence.hpp>

#include <string>

namespace cuphoto {

class Config {
private:
    cv::FileStorage file;
public: 
    ~Config();  // close the file when deconstructing

    // set a new config file
    bool set_parameter_file(const std::string& filename);

    // access the parameter values
    template <typename T>
    T get(const std::string& key, const T& default_value) const {
        cv::FileNode val = file[key];
        if (val.empty())
            return default_value;
        else
            return T(val);
    }

    template <typename T>
    T get(const std::string& key) const {
        return T(file[key]);
    }
};

}

#endif