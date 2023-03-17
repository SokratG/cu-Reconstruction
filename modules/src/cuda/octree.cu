#include "octree.cuh"
#include "CudaUtils/cudaUtility.cuh"

namespace cuphoto {

// -------------------------- Octree Node --------------------------

OctreeNode::Ptr OctreeNode::create_node(const float3 center, const r32 size, OctreeStackAllocator& osa) {
    auto node = osa.allocate();
    // placement new
    new (node) OctreeNode(center, size);
    return node;
}

OctreeNode::Ptr OctreeNode::create_node(OctreeStackAllocator& osa) {
    auto node = osa.allocate();
    new (node) OctreeNode();
    return node;
}

OctreeNode::OctreeNode() :
    _center(make_float3(0., 0., 0.)), _voxel_size(0.), _color(make_float3(0., 0., 0.)),
    _dist(-1.), _weight(0.), 
    _mean_pts(0.), _n_sample(0) {

}

OctreeNode::OctreeNode(const float3 center, const r32 size, const float3 color) : 
    _center(center), _voxel_size(size), _color(color),
    _dist(-1.), _weight(0.), 
    _mean_pts(0.), _n_sample(0) {

}

r32 OctreeNode::voxel_size() const {
    return _voxel_size;
}

void OctreeNode::voxel_size(const r32 size) {
    _voxel_size = size;
}

float3 OctreeNode::center() const {
    return _center;
}

void OctreeNode::center(const float3 center) {
    _center = center;
}
    

float3 OctreeNode::color() const {
    return _color;
}

void OctreeNode::color(const float3 color) {
    _color = color;
}

r32 OctreeNode::weight() const {
    return _weight;
}

void OctreeNode::weight(const r32 weight) {
    _weight = weight;
}

i32 OctreeNode::num_sample() const {
    return _n_sample;
}

r32 OctreeNode::mean_pts() const {
    return _mean_pts;
}

r32 OctreeNode::distance() const {
    return _dist;
}

void OctreeNode::distance(const r32 dist) {
    _dist = dist;
}

r32 OctreeNode::max_size() const {
    return factor_max_size * _voxel_size;
}

bool OctreeNode::has_children() const {
    return childs[0] != nullptr;
}


// -------------------------- Octree Stack Allocator --------------------------

OctreeStackAllocator::OctreeStackAllocator(std::size_t pool_size) : mem_pool_size(pool_size) {
    if (mem_pool_size <= 0) {
        LogError(LOG_CUDA "OctreeStackAllocator::OctreeStackAllocator() -- can't allocate query CUDA memory %lu\n", mem_pool_size);
        throw std::bad_alloc();
    }
    current_idx = 0;
    CUDA(cudaMallocManaged(&current_ptr, mem_pool_size * sizeof(OctreeNode), cudaMemAttachGlobal));
}

OctreeStackAllocator::~OctreeStackAllocator() {
    CUDA(cudaFree(current_ptr));
}

OctreeNode::Ptr OctreeStackAllocator::allocate() {
    if (current_idx > mem_pool_size) {
        LogError(LOG_CUDA "OctreeStackAllocator::allocate() -- can't allocate OctreeNode. The memory limit is exceeded. %lu\n", mem_pool_size);
        throw std::bad_alloc();
    }
    const OctreeNode::Ptr top_ptr = &current_ptr[current_idx];
    current_idx += 1;
    return top_ptr;
}

// -------------------------- Octree --------------------------
Octree::Octree() {
    
}

}