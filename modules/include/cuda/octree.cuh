#ifndef CUPHOTO_LIB_OCTREE_CUH
#define CUPHOTO_LIB_OCTREE_CUH

#include "types.cuh"

#include <memory>

namespace cuphoto {

class OctreeNode {
public:
    using Ptr = std::shared_ptr<OctreeNode>;

    OctreeNode();
};



class Octree {
public:
    using Ptr = std::shared_ptr<Octree>;
    
    Octree();
};

}

#endif

