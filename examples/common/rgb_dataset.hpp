#ifndef RGB_IMAGE_DATASET_HPP
#define RGB_IMAGE_DATASET_HPP

#include "image_dataset.hpp"
#include "types_dataset.hpp"

namespace cuphoto {


class RGBDataset : public ImageDataset<RGB> {
public:
    RGBDataset(const std::string& data_path);

    virtual RGB get_next() override;
};

}

#endif