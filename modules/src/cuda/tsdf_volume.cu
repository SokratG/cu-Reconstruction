#include "tsdf_volume.cuh"
#include "CudaUtils/cudaUtility.cuh"

namespace cuphoto {


__global__ void test_kernel(Octree::Ptr octree, ui32* count_leaves, ui32* count_voxels) {
    const i32 x = blockIdx.x * blockDim.x + threadIdx.x;
    const i32 y = blockIdx.y * blockDim.y + threadIdx.y;
    *count_leaves = octree->count_leaves();
    *count_voxels = octree->count_voxels();
}


TSDFVolume::TSDFVolume() {
    // TODO
    octree = Octree::create_octree(1000, 5.);
}

TSDFVolume::~TSDFVolume() {
    Octree::free_octree(octree);
}

bool TSDFVolume::integrate() {
    // const dim3 blockDim(1);

    octree->build_octree(2);
    ui32 count_leaves = 0; ui32 count_voxels = 0;
    ui32* cu_count_leaves; ui32* cu_count_voxels;

    CUDA(cudaMalloc(&cu_count_leaves, sizeof(ui32)));
    CUDA(cudaMalloc(&cu_count_voxels, sizeof(ui32)));

    CUDA(cudaMemset(cu_count_leaves, 0, sizeof(ui32)));
    CUDA(cudaMemset(cu_count_voxels, 0, sizeof(ui32)));

    test_kernel<<<1, 1>>>(octree, cu_count_leaves, cu_count_voxels);
    CUDA(cudaMemcpy(&count_leaves, cu_count_leaves, sizeof(ui32), cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(&count_voxels, cu_count_voxels, sizeof(ui32), cudaMemcpyDeviceToHost));

    LogInfo(LOG_CUDA "COUNT LEAVES %u\n", count_leaves); // must eq. 64
    LogInfo(LOG_CUDA "COUNT VOXELS %u\n", count_voxels); // must eq. 73

   

    CUDA(cudaDeviceSynchronize());

    return true;
}

}