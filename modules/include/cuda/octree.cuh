#ifndef CUPHOTO_LIB_OCTREE_CUH
#define CUPHOTO_LIB_OCTREE_CUH

#include "types.cuh"
#include "object_pool_allocator.cuh"

#include <vector>
#include <memory>

namespace cuphoto {

#define INSIDE_VIEW 1
#define OUTSIDE_VIEW -INSIDE_VIEW

class OctreeNode {
public:
    using Ptr = OctreeNode*;

    struct OctreeListNode {
        using Ptr = OctreeListNode*;
        using PPtr = OctreeListNode**;
        using Ref = OctreeListNode&;
        using PRef = OctreeListNode::Ptr&;

        OctreeNode::Ptr oct_node {nullptr};
        OctreeListNode::Ptr next {nullptr};
        i64 n_level = 0;
    };

    static constexpr auto OCT_NUM = 8;
    static constexpr r32 factor_max_size = 1.732050807;

    OctreeNode::Ptr childs[OCT_NUM] {nullptr};
public:
    __host__ __device__ static OctreeNode* create_node(const float3 center, const r32 voxel_size,
                                                       CudaObjectPoolAllocator<OctreeNode>::Ref oct_pa);
    __host__ __device__ static OctreeNode* create_node(CudaObjectPoolAllocator<OctreeNode>::Ref oct_pa);

    __host__ __device__ r32 voxel_size() const;

    __host__ __device__ float3 center() const;

    __host__ __device__ float3 color() const;
    __host__ __device__ void color(const float3 color);

    __host__ __device__ r32 weight() const;
    __host__ __device__ void weight(const r32 weight);

    __host__ __device__ i32 num_sample() const;
    
    __host__ __device__ r32 weighted_mean() const;

    __host__ __device__ r32 distance() const;
    __host__ __device__ void distance(const r32 dist);

    __host__ __device__ bool in_view() const;
    __host__ __device__ void in_view(const i32 inside_view /*1 inside, other outside*/);

    __host__ __device__ r32 max_size() const;

    __host__ __device__ OctreeNode::Ptr query_voxel(const float3 pt, const r32 query_size);

    __host__ __device__ void add_observation(const r32 dist, const r32 weight, const r32 max_weight, const float3 rgb);

    __host__ __device__ bool has_children() const;

    __host__ __device__ bool variance(r32& variance_value, const i32 min_sample = 5) const;

    __host__ __device__ OctreeNode** split(CudaObjectPoolAllocator<OctreeNode>::Ref oct_pa);

    __host__ __device__ void split_by_level(CudaObjectPoolAllocator<OctreeNode>::Ref oct_pa, const i32 n_level);

    __host__ __device__ void update_avg();

    __host__ __device__ void leaves(OctreeNode::OctreeListNode::PRef oct_list_head, const i32 n_level,
                                    CudaObjectPoolAllocator<OctreeListNode>::Ref oct_list_pa);
    
    __host__ __device__ void leaves(OctreeNode::OctreeListNode::PRef oct_list_head, 
                                    CudaObjectPoolAllocator<OctreeListNode>::Ref oct_list_pa);

    __host__ __device__ void count_leaves(ui32& count);

    __host__ __device__ ui32 count_voxels();

protected:
    OctreeNode();

    OctreeNode(const float3 center, const r32 size, const float3 color = make_float3(255., 255., 255.));

    __host__ __device__ void center(const float3 center);

    __host__ __device__ void voxel_size(const r32 size);

private:
    __host__ __device__ void number_voxels(ui32& count);

    __host__ __device__ OctreeNode::OctreeListNode* collect_leaves(OctreeNode::OctreeListNode::Ptr oct_list_tail,
                                                                   OctreeNode::OctreeListNode::PRef oct_list_head, 
                                                                   const i32 n_level,
                                                                   CudaObjectPoolAllocator<OctreeListNode>::Ref oct_pa);
private:
    float3 _center;
    float3 _color;
    r32 _weight;
    i32 _n_sample;
    r32 _weighted_mean;
    r32 _dist;
    r32 _voxel_size;
    i32 _in_view;
};




class Octree : CudaManaged {
public:
    using Ptr = Octree*;

    static constexpr r32 split_factor = 0.693147;
public:
    __host__ static Octree* create_octree(const r32 resolution_size);
    __host__ static Octree* create_octree(const ui32 octree_pool_size, const r32 resolution_size);
    __host__ static void free_octree(Octree* octree);

    __host__ void reserve_pool(const ui32 octree_pool_size);

    __host__ __device__ r32 resolution_size() const;

    __host__ bool build_octree(const i32 n_level_split, const float3 center = make_float3(0., 0., 0.));
    __host__ bool build_octree(const ui32 octree_pool_size, const i32 n_level_split, const float3 center = make_float3(0., 0., 0.));
    __host__ bool build_octree(const float3 max_resolution_size, const float3 center = make_float3(0., 0., 0.));
    __host__ bool build_octree(const ui32 octree_pool_size, const float3 max_resolution_size, const float3 center = make_float3(0., 0., 0.));

    __host__ __device__ static i32 desire_octree_levels(const float3 max_resolution_size, const r32 voxel_resolution_size);

    __host__ __device__ OctreeNode::Ptr root() const;

    __host__ __device__ OctreeNode::Ptr query_voxel(const float3 pt, const r32 query_size = -1.);

    __host__ __device__ bool leaves(OctreeNode::OctreeListNode::PRef oct_list_head, const i32 n_level,
                                    CudaObjectPoolAllocator<OctreeNode::OctreeListNode>::Ref oct_list_pa);
    
    __host__ __device__ bool leaves(OctreeNode::OctreeListNode::PRef oct_list_head,
                                    CudaObjectPoolAllocator<OctreeNode::OctreeListNode>::Ref oct_list_pa);

    __host__ __device__ ui32 count_leaves() const;

    __host__ __device__ ui32 count_voxels() const;

    ~Octree();
protected:

    Octree(const r32 resolution_size);

    Octree(const ui32 octree_pool_size, const r32 resolution_size);

    __host__ void resolution_size(const r32 resolution_size);
private:
    r32 _voxel_resolution_size;

    OctreeNode::Ptr octree_root;
    CudaObjectPoolAllocator<OctreeNode> oct_node_pa;
};

}

#endif

