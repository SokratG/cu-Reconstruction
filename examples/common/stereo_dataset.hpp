#ifndef STEREO_IMAGE_DATASET_HPP
#define STEREO_IMAGE_DATASET_HPP

#include "image_dataset.hpp"
#include "types_dataset.hpp"


namespace cuphoto {

class StereoDataset : public ImageDataset<STEREO> {
public:
    StereoDataset(const std::string& data_path);

    virtual STEREO get_next() override;
    virtual i32 num_files() const override;
};

}


#endif