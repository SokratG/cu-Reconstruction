#ifndef RGBD_IMAGE_DATASET_HPP
#define RGBD_IMAGE_DATASET_HPP

#include "image_dataset.hpp"
#include "types_dataset.hpp"


namespace curec {

class RGBDDataset : public ImageDataset<RGBD> {
public:
    RGBDDataset(const std::string& data_path);

    virtual RGBD get_next() override;
    virtual i32 num_files() const override;
};

}


#endif