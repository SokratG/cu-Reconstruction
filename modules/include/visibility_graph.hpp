#ifndef CUREC_LIB_VISIBILITY_GRAPH_HPP
#define CUREC_LIB_VISIBILITY_GRAPH_HPP

#include "feature_matcher.hpp"
#include "landmark.hpp"
#include "utils.hpp"
#include <unordered_map>

namespace curec {

class VisibilityGraph {
public:
    VisibilityGraph();

    void build_nodes(const std::vector<MatchAdjacent>& ma, 
                     const std::vector<std::vector<Feature>>& feat_pts);
private:
    std::unordered_map<uuid, Landmark> graph;
};

};


#endif