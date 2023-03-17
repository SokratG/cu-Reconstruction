#ifndef CUPHOTO_LIB_OCTREE_CUH
#define CUPHOTO_LIB_OCTREE_CUH

#include "types.cuh"

#include <vector>
#include <memory>

namespace cuphoto {

struct OctreeStackAllocator;

class OctreeNode {
public:
    using Ptr = OctreeNode*;

    static constexpr auto OCT_NUM = 8;
    static constexpr r32 factor_max_size = 1.732050807;

    OctreeNode::Ptr childs[OCT_NUM] {nullptr};
public:
    static OctreeNode* create_node(const float3 center, const r32 size,
                                   OctreeStackAllocator& osa);
    static OctreeNode* create_node(OctreeStackAllocator& osa);

    r32 voxel_size() const;

    float3 center() const;

    __host__ __device__ float3 color() const;
    __host__ __device__ void color(const float3 color);

    r32 weight() const;
    void weight(const r32 weight);

    i32 num_sample() const;
    
    r32 mean_pts() const;

    r32 distance() const;
    void distance(const r32 dist);

    r32 max_size() const;

    bool has_children() const;

protected:
    OctreeNode();

    OctreeNode(const float3 center, const r32 size, const float3 color = make_float3(255., 255., 255.));

    void center(const float3 center);

    void voxel_size(const r32 size);

private:
    float3 _center;
    float3 _color;
    r32 _weight;
    i32 _n_sample;
    r32 _mean_pts;
    r32 _dist;
    r32 _voxel_size;
};


struct OctreeStackAllocator {
public:
    const std::size_t mem_pool_size;
public:
    OctreeStackAllocator(std::size_t pool_size);

    ~OctreeStackAllocator();

    OctreeNode::Ptr allocate();

protected:
    friend class OctreeNode;
    OctreeNode::Ptr current_ptr = nullptr;
    ui64 current_idx;
};


class Octree {
public:
    using Ptr = Octree*;
    
    Octree();
private:
    OctreeNode::Ptr octree_head;
};

}

#endif

