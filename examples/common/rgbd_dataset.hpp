#ifndef RGBD_IMAGE_DATASET_HPP
#define RGBD_IMAGE_DATASET_HPP

#include "image_dataset.hpp"
#include "types_dataset.hpp"


namespace curec {

class RGBDDataset : public ImageDataset<RGBD> {
public:
    RGBDDataset(const std::string& data_path, const r64 depth_scale);

    virtual RGBD get_next() override;
    virtual i32 num_files() const override;
private:
    r64 depth_scale;
};

}


#endif