#include "visibility_graph.hpp"
#include <opencv2/core/eigen.hpp>

namespace curec {

VisibilityGraph::VisibilityGraph(const r64 _init_point_depth, const bool _use_color_landmark) 
                                : init_point_depth(_init_point_depth), use_color_landmark(_use_color_landmark)
{

}


Landmark::Ptr VisibilityGraph::make_landmark(const KeyFrame::Ptr frame, const Feature::Ptr feat_pt) 
{
    const auto pt2 = feat_pt->position.pt;
    const Vec3f color = cv_rgb_2_eigen_rgb(frame->frame().at<cv::Vec3b>(pt2));
    const Vec3 position = Vec3(pt2.x, pt2.y, init_point_depth);
    Landmark::Ptr landmark = Landmark::create_landmark(position, color);
    landmark->set_observation(feat_pt);
    return landmark;
}

void VisibilityGraph::build_nodes(const std::vector<KeyFrame::Ptr>& frames,
                                  const std::vector<MatchAdjacent>& ma,
                                  const std::vector<std::vector<Feature::Ptr>>& feat_pts) 
{
    std::unordered_map<ui64, ui64> localVisibilityGraph;
    ui64 vertices_index = 0;
    for (const auto& adj : ma) {
        const i32 src_frame = adj.src_idx;
        const i32 dst_frame = adj.dst_idx;
        for (const auto& match : adj.match) {
            const ui64 p1_idx = match.queryIdx; 
            const ui64 p2_idx = match.trainIdx;

            const ui64 key1 = gen_combined_key(src_frame, p1_idx); 
            const ui64 key2 = gen_combined_key(dst_frame, p2_idx);
            const auto visitView1 = localVisibilityGraph.find(key1);
            const auto visitView2 = localVisibilityGraph.find(key2);
            if (visitView1 != localVisibilityGraph.end() && visitView2 != localVisibilityGraph.end()) {
                
                // remove previous view points
                if (visitView1->second != visitView2->second) {
                    localVisibilityGraph.erase(visitView1);
                    localVisibilityGraph.erase(visitView2);
                }
                continue;
            } 
            
            ui64 visibility_idx = 0;
            if (visitView1 != localVisibilityGraph.end())
                visibility_idx = visitView1->second;
            else if (visitView2 != localVisibilityGraph.end())
                visibility_idx = visitView2->second;
            else {
                // add new landmark for current view 
                visibility_idx = vertices_index;
                vertices_index += 1;
                auto feat_pt = feat_pts[src_frame].at(p1_idx);
                feat_pt->frame = frames[src_frame];
                Landmark::Ptr landmark = make_landmark(frames[src_frame], feat_pt);
                vis_graph.insert({key1, landmark});
            }

            if (visitView1 == localVisibilityGraph.end())
                localVisibilityGraph[key1] = visibility_idx;
            if (visitView2 == localVisibilityGraph.end())
                localVisibilityGraph[key2] = visibility_idx;  
        }
    }

}


};