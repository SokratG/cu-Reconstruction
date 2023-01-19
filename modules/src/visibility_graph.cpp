#include "visibility_graph.hpp"
#include "motion_estimation.hpp"
#include "utils.hpp"

namespace curec {

void build_landmarks_graph_triangluation(const std::vector<MatchAdjacent>& matching,
                                         const std::vector<KeyFrame::Ptr>& frames,
                                         const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                         const Camera::Ptr camera,
                                         VisibilityGraph& vis_graph,
                                         std::vector<Landmark::Ptr>& landmarks) 
{
    // TODO: add config
    vis_graph = VisibilityGraph();
    landmarks = std::vector<Landmark::Ptr>();
    for (const auto& adj : matching) {
        const i32 src_idx = adj.src_idx;
        if (adj.dst_idx < 0)
            continue;
        const i32 dst_idx = adj.dst_idx;
        const auto& match = adj.match;
        const auto match_size = match.size();

        for (auto i = 0; i < match_size; ++i) {
            const auto pt_src_idx = match.at(i).queryIdx; 
            const auto pt_dst_idx = match.at(i).trainIdx;
            
            const auto src_feat_pt = feat_pts[src_idx].at(pt_src_idx);
            const auto dst_feat_pt = feat_pts[dst_idx].at(pt_dst_idx);

            const auto src_pt = src_feat_pt->position.pt;
            const auto dst_pt = dst_feat_pt->position.pt;

            const auto key1 = gen_combined_key(src_idx, pt_src_idx); 
            const auto key2 = gen_combined_key(dst_idx, pt_dst_idx); 

            auto visitView1 = vis_graph.find(key1);
            auto visitView2 = vis_graph.find(key2);

            if (visitView1 != vis_graph.end() && visitView2 != vis_graph.end()) {
                
                // remove previous view points
                if (visitView1->second->landmark_idx() != visitView2->second->landmark_idx()) {
                    vis_graph.erase(visitView1);
                    vis_graph.erase(visitView2);
                }
                continue;
            }
            
            Vec3 pt_world = Vec3::Zero();
            const auto pt_camera = project_px_point(camera, src_pt, dst_pt);
            const bool is_valid = triangulation(frames.at(src_idx)->pose(), frames.at(dst_idx)->pose(),
                                                pt_camera, 1e-2, pt_world);
            ui32 visibility_idx = 0;
            if (visitView1 != vis_graph.end())
                visibility_idx = visitView1->second->landmark_idx();
            else if (visitView2 != vis_graph.end())
                visibility_idx = visitView2->second->landmark_idx();
            else {
                visibility_idx = landmarks.size();
                if (is_valid) {
                    Landmark::Ptr landmark = make_landmark(dst_feat_pt, pt_world);
                    landmarks.emplace_back(landmark);
                }
            }
            
            if (is_valid) {
                if (visitView1 == vis_graph.end())
                    vis_graph[key1] = std::make_shared<VisibilityNode>(src_idx, pt_src_idx, visibility_idx);
                if (visitView2 == vis_graph.end())
                    vis_graph[key2] = std::make_shared<VisibilityNode>(dst_idx, pt_dst_idx, visibility_idx);
            }
        }    
    }
}

void build_landmarks_graph_depth(const std::vector<MatchAdjacent>& matching,
                                 const std::vector<KeyFrame::Ptr>& frames,
                                 const std::vector<cv::Mat>& depth,
                                 const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                 const Camera::Ptr camera,
                                 VisibilityGraph& vis_graph,
                                 std::vector<Landmark::Ptr>& landmarks)
{
    vis_graph = VisibilityGraph();
    landmarks = std::vector<Landmark::Ptr>();
    const auto K = camera->K();
    for (const auto& adj : matching) {
        const i32 src_idx = adj.src_idx;
        if (adj.dst_idx < 0)
            continue;
        const i32 dst_idx = adj.dst_idx;
        const auto& match = adj.match;
        const auto match_size = match.size();

        for (auto i = 0; i < match_size; ++i) {
            const auto pt_src_idx = match.at(i).queryIdx; 
            const auto pt_dst_idx = match.at(i).trainIdx;
            
            const auto src_feat_pt = feat_pts[src_idx].at(pt_src_idx);
            const auto dst_feat_pt = feat_pts[dst_idx].at(pt_dst_idx);

            const auto src_pt = src_feat_pt->position.pt;
            const auto dst_pt = dst_feat_pt->position.pt;

            const auto key1 = gen_combined_key(src_idx, pt_src_idx); 
            const auto key2 = gen_combined_key(dst_idx, pt_dst_idx); 

            auto visitView1 = vis_graph.find(key1);
            auto visitView2 = vis_graph.find(key2);

            if (visitView1 != vis_graph.end() && visitView2 != vis_graph.end()) {
                
                // remove previous view points
                if (visitView1->second->landmark_idx() != visitView2->second->landmark_idx()) {
                    vis_graph.erase(visitView1);
                    vis_graph.erase(visitView2);
                }
                continue;
            }
            
            const r64 z = depth.at(src_idx).at<r32>(src_pt);
            const r64 x = (src_pt.x - K(0, 2)) * z / K(0, 0);
            const r64 y = (src_pt.y - K(1, 2)) * z / K(1, 1);
            Vec3 pt_world(x, y, z);
            
            ui32 visibility_idx = 0;
            if (visitView1 != vis_graph.end())
                visibility_idx = visitView1->second->landmark_idx();
            else if (visitView2 != vis_graph.end())
                visibility_idx = visitView2->second->landmark_idx();
            else {
                if (pt_world.z() > 1e-7) {
                    visibility_idx = landmarks.size();
                    Landmark::Ptr landmark = make_landmark(dst_feat_pt, pt_world);
                    landmarks.emplace_back(landmark);
                }   
            }
            
            if (pt_world.z() > 1e-7) {
                if (visitView1 == vis_graph.end())
                    vis_graph[key1] = std::make_shared<VisibilityNode>(src_idx, pt_src_idx, visibility_idx);
                if (visitView2 == vis_graph.end())
                    vis_graph[key2] = std::make_shared<VisibilityNode>(dst_idx, pt_dst_idx, visibility_idx);
            }
        }    
    }

}


};