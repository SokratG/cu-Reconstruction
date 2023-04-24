#include "octree.cuh"


namespace cuphoto {

// -------------------------- Octree Node --------------------------

OctreeNode::Ptr OctreeNode::create_node(const float3 center, const r32 voxel_size, 
                                        CudaObjectPoolAllocator<OctreeNode>::Ref oct_pa) {
    auto node = oct_pa.allocate();
    // placement new ?
    // new (node) OctreeNode(center, voxel_size);
    node->center(center);
    node->voxel_size(voxel_size);
    return node;
}

OctreeNode::Ptr OctreeNode::create_node(CudaObjectPoolAllocator<OctreeNode>::Ref oct_pa) {
    auto node = oct_pa.allocate();
    return node;
}

OctreeNode::OctreeNode() :
    _center(make_float3(0., 0., 0.)), _voxel_size(0.), _color(make_float3(0., 0., 0.)),
    _dist(-1.), _weight(0.), _in_view(OUTSIDE_VIEW),
    _weighted_mean(0.), _n_sample(0) {

}

OctreeNode::OctreeNode(const float3 center, const r32 size, const float3 color) : 
    _center(center), _voxel_size(size), _color(color),
    _dist(-1.), _weight(0.), _in_view(OUTSIDE_VIEW),
    _weighted_mean(0.), _n_sample(0) {

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

r32 OctreeNode::weighted_mean() const {
    return _weighted_mean;
}

r32 OctreeNode::distance() const {
    return _dist;
}

void OctreeNode::distance(const r32 dist) {
    _dist = dist;
}


bool OctreeNode::in_view() const {
    return _in_view == INSIDE_VIEW; 
}


void OctreeNode::in_view(const i32 inside_view) {
    _in_view = inside_view;
}

r32 OctreeNode::max_size() const {
    return factor_max_size * _voxel_size;
}

bool OctreeNode::has_children() const {
    for (i32 idx = 0; idx < OCT_NUM; ++idx) {
        if (childs[idx] != nullptr)
            return true;
    }
    return false;
}

OctreeNode::Ptr OctreeNode::query_voxel(const float3 pt, const r32 query_size) {
    const bool has_child = has_children();
    const bool contain = query_size > 0 && _voxel_size <= query_size;
    if (!has_child || contain) {
        return this;
    } else {
        const i32 child_idx = ((pt.x - _center.x) > 0) * 4 + ((pt.y - _center.y) > 0) * 2 + ((pt.z - _center.z) > 0);
        return childs[child_idx]->query_voxel(pt, query_size);
    }
}

void OctreeNode::add_observation(const r32 dist, const r32 weight, const r32 max_weight, const float3 rgb) {
    const r32 total_weight = _weight + weight;
    _color.x = (_weight * _color.x + weight * rgb.x) / total_weight;
    _color.y = (_weight * _color.y + weight * rgb.y) / total_weight;
    _color.z = (_weight * _color.z + weight * rgb.z) / total_weight;
    const r32 prev_dist = _dist;
    _dist = (_dist / _weight + dist * weight) / total_weight;
    _weight += weight;
    if (_weight > max_weight)
        _weight = max_weight;
    _weighted_mean += weight * (dist - _dist) * (dist - prev_dist);
    _n_sample += 1;
}

bool OctreeNode::variance(r32& variance_value, const i32 min_sample) const {
    if (_n_sample < min_sample)
        return false;
    variance_value = (_weighted_mean / _weight) * (_n_sample / (_n_sample - 1));
    return true;
}

OctreeNode** OctreeNode::split(CudaObjectPoolAllocator<OctreeNode>::Ref oct_pa) {
    const r32 voxel_half_size = _voxel_size / 2;
    const r32 voxel_offset_x = _voxel_size / 4;
    const r32 voxel_offset_y = _voxel_size / 4;
    const r32 voxel_offset_z = _voxel_size / 4;
    float3 center = make_float3(_center.x - voxel_offset_x, _center.y - voxel_offset_y, _center.z - voxel_offset_z);
    childs[0] = create_node(center, voxel_half_size, oct_pa);
    center = make_float3(_center.x - voxel_offset_x, _center.y - voxel_offset_y, _center.z + voxel_offset_z);
    childs[1] = create_node(center, voxel_half_size, oct_pa);
    center = make_float3(_center.x - voxel_offset_x, _center.y + voxel_offset_y, _center.z - voxel_offset_z);
    childs[2] = create_node(center, voxel_half_size, oct_pa);
    center = make_float3(_center.x - voxel_offset_x, _center.y + voxel_offset_y, _center.z + voxel_offset_z);
    childs[3] = create_node(center, voxel_half_size, oct_pa);
    center = make_float3(_center.x + voxel_offset_x, _center.y - voxel_offset_y, _center.z - voxel_offset_z);
    childs[4] = create_node(center, voxel_half_size, oct_pa);
    center = make_float3(_center.x + voxel_offset_x, _center.y - voxel_offset_y, _center.z + voxel_offset_z);
    childs[5] = create_node(center, voxel_half_size, oct_pa);
    center = make_float3(_center.x + voxel_offset_x, _center.y + voxel_offset_y, _center.z - voxel_offset_z);
    childs[6] = create_node(center, voxel_half_size, oct_pa);
    center = make_float3(_center.x + voxel_offset_x, _center.y + voxel_offset_y, _center.z + voxel_offset_z);
    childs[7] = create_node(center, voxel_half_size, oct_pa);
    return childs;
}

void OctreeNode::split_by_level(CudaObjectPoolAllocator<OctreeNode>::Ref oct_pa, const i32 n_level) {
    if (n_level <= 0)
        return;
    this->split(oct_pa);
    for (i32 idx = 0; idx < OCT_NUM; ++idx) {
        if (childs[idx] != nullptr) {
            childs[idx]->split_by_level(oct_pa, n_level - 1);
        }
    }
}

void OctreeNode::update_avg() {
    if (!has_children())
        return;

    r32 dist_average = 0.0;
    r32 weight_average = 0.0;
    i32 n_sample = 0;
    for (i32 idx = 0; idx < OCT_NUM; ++idx) {
        if (childs[idx] != nullptr) {
            dist_average += childs[idx]->distance();
            weight_average += childs[idx]->weight();
            n_sample += 1;
        }
    }

    if (n_sample > 0) {
        const r32 dist = dist_average / n_sample;
        const r32 weight = weight_average / n_sample;
        this->distance(dist);
        this->weight(weight);
    }
}


void OctreeNode::leaves(OctreeNode::OctreeListNode::Ptr& oct_list_head, const i32 n_level, 
                        CudaObjectPoolAllocator<OctreeListNode>::Ref oct_pa) {
    if (n_level <= 0)
        return;
    OctreeNode::OctreeListNode::Ptr oct_list_tail = nullptr;
    collect_leaves(oct_list_tail, oct_list_head, n_level, oct_pa);
}

void OctreeNode::leaves(OctreeNode::OctreeListNode::PRef oct_list_head, CudaObjectPoolAllocator<OctreeListNode>::Ref oct_pa) {
    OctreeNode::OctreeListNode::Ptr oct_list_tail = nullptr;
    constexpr i32 deepest_level = -1;
    collect_leaves(oct_list_tail, oct_list_head, deepest_level, oct_pa);
}

OctreeNode::OctreeListNode::Ptr OctreeNode::collect_leaves(OctreeNode::OctreeListNode::Ptr oct_list_tail,
                                                           OctreeNode::OctreeListNode::Ptr& oct_list_head,
                                                           const i32 n_level, CudaObjectPoolAllocator<OctreeListNode>::Ref oct_pa) {
    if (!this->has_children()) {
        if (oct_list_tail == nullptr) {
            // init list head
            oct_list_tail = oct_pa.allocate();
            oct_list_tail->oct_node = this;
            oct_list_head = oct_list_tail;
        } else {
            OctreeListNode::Ptr curr = oct_pa.allocate();
            curr->oct_node = this;
            oct_list_tail->next = curr;
            oct_list_tail = curr;
        }
        oct_list_tail->n_level = n_level;
    } else {
        for (i32 idx = 0; idx < OCT_NUM; ++idx) {
            const OctreeNode::Ptr child = childs[idx];
            if (child != nullptr && n_level != 0) {
                oct_list_tail = child->collect_leaves(oct_list_tail, oct_list_head, n_level - 1, oct_pa);
            }
        }
    }
    return oct_list_tail;
}

void OctreeNode::count_leaves(ui32& count) {
    if (!this->has_children()) {
        count += 1;
    } else {
        for (i32 idx = 0; idx < OCT_NUM; ++idx) {
            const OctreeNode::Ptr child = childs[idx];
            if (child != nullptr) {
                child->count_leaves(count);
            }
        }
    }
}

ui32 OctreeNode::count_voxels() {
    ui32 count = 0;
    number_voxels(count);
    return count;
}

void OctreeNode::number_voxels(ui32& count) {
    count += 1;
    for (i32 idx = 0; idx < OCT_NUM; ++idx) {
        const OctreeNode::Ptr child = childs[idx];
        if (child != nullptr) {
            if (child->has_children()) {
                child->number_voxels(count);
            } else {
                count += 1;
            }
        }
    }
}

// -------------------------- Octree --------------------------
Octree::Octree(const r32 resolution_size) : 
               _voxel_resolution_size(resolution_size), octree_root(nullptr) {
}

Octree::Octree(const ui32 octree_pool_size, const r32 resolution_size) :
               _voxel_resolution_size(resolution_size), octree_root(nullptr) {
    oct_node_pa.reserve(octree_pool_size);
}


Octree::Ptr Octree::create_octree(const r32 resolution_size) {
    Octree::Ptr octree = new Octree(resolution_size);
    return octree;
}

Octree::Ptr Octree::create_octree(const ui32 octree_pool_size, const r32 resolution_size) {
    Octree::Ptr octree = new Octree(octree_pool_size, resolution_size);
    return octree;
}

Octree::~Octree() {

}

void Octree::free_octree(Octree::Ptr octree) {
    delete octree;
}

void Octree::reserve_pool(const ui32 octree_pool_size) {
    oct_node_pa.reserve(octree_pool_size);
}

r32 Octree::resolution_size() const {
    return _voxel_resolution_size;
}

void Octree::resolution_size(const r32 resolution_size) {
    _voxel_resolution_size = resolution_size;
}


OctreeNode::Ptr Octree::root() const {
    return octree_root;
}

bool Octree::build_octree(const i32 n_level_split, const float3 center) {
    if (oct_node_pa.pool_size() <= 0) {
        LogWarning(LOG_CUDA "Octree::build_octree() -- can't build octree nodes, the object pool was not initialized\n");
        return false;
    }
    octree_root = OctreeNode::create_node(center, _voxel_resolution_size, oct_node_pa);
    if (octree_root == nullptr) {
        LogWarning(LOG_CUDA "Octree::build_octree() -- can't init octree root\n");
        return false;
    }
    octree_root->split_by_level(oct_node_pa, n_level_split);
    return true;
}

bool Octree::build_octree(const float3 max_resolution_size, const float3 center) {
    const i32 n_split_level =  desire_octree_levels(max_resolution_size, _voxel_resolution_size);
    return build_octree(n_split_level, center);
}


bool Octree::build_octree(const ui32 octree_pool_size, const i32 n_level_split, const float3 center) {
    reserve_pool(octree_pool_size);
    return build_octree(n_level_split, center);
}

bool Octree::build_octree(const ui32 octree_pool_size, const float3 max_resolution_size, const float3 center) {
    reserve_pool(octree_pool_size);
    const i32 n_split_level =  desire_octree_levels(max_resolution_size, _voxel_resolution_size);
    return build_octree(n_split_level, center);
}


i32 Octree::desire_octree_levels(const float3 max_resolution_size, const r32 voxel_resolution_size) {
    const r32 desired_resol = fmaxf(voxel_resolution_size / max_resolution_size.x, 
                                    fmaxf(voxel_resolution_size / max_resolution_size.y, 
                                          voxel_resolution_size / max_resolution_size.z));
    i32 n_level = ceil(logf(desired_resol) / split_factor);
    return n_level;
}


OctreeNode::Ptr Octree::query_voxel(const float3 pt, const r32 query_size) {
    if (octree_root == nullptr)
        return nullptr;

    const r32 half_voxel_resolution = _voxel_resolution_size / 2;
    if (fabsf(pt.x) > half_voxel_resolution || 
        fabsf(pt.y) > half_voxel_resolution || 
        fabsf(pt.z) > half_voxel_resolution) {
        return nullptr;
    }
    OctreeNode::Ptr voxel = octree_root->query_voxel(pt, query_size); // check?
    return voxel;
}

bool Octree::leaves(OctreeNode::OctreeListNode::PRef oct_list_head, const i32 n_level,
                    CudaObjectPoolAllocator<OctreeNode::OctreeListNode>::Ref oct_list_pa) {
    if (octree_root == nullptr || oct_list_pa.pool_size() <= 0) {
        oct_list_head = nullptr;
        return false;
    }

    if (n_level == 0) {
        oct_list_head = oct_list_pa.allocate();
        oct_list_head->oct_node = octree_root;
    } else {
        octree_root->leaves(oct_list_head, n_level, oct_list_pa);
    }
    return true;
}

bool Octree::leaves(OctreeNode::OctreeListNode::PRef oct_list_head,
                    CudaObjectPoolAllocator<OctreeNode::OctreeListNode>::Ref oct_list_pa) {
    if (octree_root == nullptr || oct_list_pa.pool_size() <= 0) {
        oct_list_head = nullptr;
        return false;
    }
    octree_root->leaves(oct_list_head, oct_list_pa);
    return true;
}

ui32 Octree::count_leaves() const {
    ui32 count = 0;
    if (octree_root == nullptr)
        return count;
    
    octree_root->count_leaves(count);
    return count;
}

ui32 Octree::count_voxels() const {
    if (octree_root == nullptr)
        return 0;
    return octree_root->count_voxels();
}

}