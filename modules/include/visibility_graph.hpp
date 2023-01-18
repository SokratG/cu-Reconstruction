#ifndef CUREC_LIB_VISIBILITY_GRAPH_HPP
#define CUREC_LIB_VISIBILITY_GRAPH_HPP

#include "types.hpp"

namespace curec {

class VisibilityNode
{
public:
    using Ptr = std::shared_ptr<VisibilityNode>;

    VisibilityNode() {}
    VisibilityNode(const ui32 frame_idx, const ui32 obs_idx, const ui32 landmark_idx) :
                    frame_idx_(frame_idx), obs_idx_(obs_idx), landmark_idx_(landmark_idx) { }

    bool operator==(const VisibilityNode& rhs) {
        return (frame_idx_ == rhs.frame_idx_) && (obs_idx_ == rhs.obs_idx_) && (landmark_idx_ == rhs.landmark_idx_);
    }
    bool operator!=(const VisibilityNode& rhs) {
        return !((*this) == rhs);
    }

    ui32 frame_idx() const {return frame_idx_;}
    ui32 frame_idx(const ui32 frame_idx) {frame_idx_=frame_idx; return frame_idx_;}

    ui32 obs_idx() const {return obs_idx_;}
    ui32 obs_idx(const ui32 obs_idx) {obs_idx_=obs_idx; return obs_idx_;}

    ui32 landmark_idx() const {return landmark_idx_;}
    ui32 landmark_idx(const ui32 landmark_idx) {landmark_idx_=landmark_idx; return landmark_idx_;}

private:
    ui32 frame_idx_;
    ui32 obs_idx_;
    ui32 landmark_idx_;
};


using VisibilityGraph = std::unordered_map<ui64, VisibilityNode::Ptr>;

};


#endif