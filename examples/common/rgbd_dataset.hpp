#ifndef RGBD_IMAGE_DATASET_HPP
#define RGBD_IMAGE_DATASET_HPP

#include "image_dataset.hpp"
#include <opencv2/core/cuda.hpp>
#include <tuple>

namespace curec {

using RGB = cv::cuda::GpuMat;
using DEPTH = cv::cuda::GpuMat;
using RGBD = std::tuple<RGB, DEPTH>;

class RGBDDataset : public ImageDataset<RGBD> {
public:
    RGBDDataset(const std::string& data_path);

    virtual RGBD get_next() override;
    virtual i32 num_files() const override;
    void reset();
    void set_data_directory(const std::string& data_path);
private:
    std::set<std::string> files;
    std::set<std::string>::iterator current_file;
};

}


#endif