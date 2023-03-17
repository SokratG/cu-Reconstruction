#include "tsdf_volume.cuh"
#include "CudaUtils/cudaUtility.cuh"

namespace cuphoto {


TSDFVolume::TSDFVolume() : osa(100) {
    
}

bool TSDFVolume::integrate() {
    // const dim3 blockDim(1);
    
    head = OctreeNode::create_node(make_float3(1, 2, 3), 5, osa);

    CUDA(cudaDeviceSynchronize());

    return true;
}

}