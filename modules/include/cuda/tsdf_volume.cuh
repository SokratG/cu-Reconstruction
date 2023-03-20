#ifndef CUPHOTO_LIB_TSDF_VOLUME_CUH
#define CUPHOTO_LIB_TSDF_VOLUME_CUH

#include "types.cuh"
#include "octree.cuh"
#include "cuda_point_cloud.cuh"

#include <memory>

namespace cuphoto {


class TSDFVolume {
public:
    using Ptr = std::shared_ptr<TSDFVolume>;

    TSDFVolume();
    ~TSDFVolume();

    bool integrate();
public:
    Octree::Ptr octree;
};

}


#endif