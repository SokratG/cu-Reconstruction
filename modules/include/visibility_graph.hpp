#ifndef CUREC_LIB_VISIBILITY_GRAPH_HPP
#define CUREC_LIB_VISIBILITY_GRAPH_HPP

#include "keyframe.hpp"
#include "feature_matcher.hpp"
#include "landmark.hpp"
#include "utils.hpp"
#include <memory>
#include <unordered_map>

namespace curec {

class VisibilityGraph {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<VisibilityGraph>;

    VisibilityGraph(const r64 init_point_depth, const bool use_color_landmark = false);
    VisibilityGraph() = delete;

    void build_nodes(const std::vector<KeyFrame::Ptr>& frames,
                     const std::vector<MatchAdjacent>& ma, 
                     const std::vector<std::vector<Feature::Ptr>>& feat_pts);
public:
    std::unordered_map<ui64, Landmark::Ptr> vis_graph;
private:
    Landmark::Ptr make_landmark(const KeyFrame::Ptr frame, const Feature::Ptr feat_pt);
private:
    const r64 init_point_depth;
    const bool use_color_landmark;
};

};


#endif