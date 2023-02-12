#ifndef CUPHOTO_LIB_SFM_HPP
#define CUPHOTO_LIB_SFM_HPP

#include "visibility_graph.hpp"
#include <string_view>
#include <vector>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/core.hpp>


namespace cuphoto {


struct MatchAdjacent;

class Sfm {
public:
    Sfm(const Camera::Ptr camera);
    bool add_frame(const cv::cuda::GpuMat frame);
    void run_pipeline();
    void store_to_ply(const std::string_view& ply_filepath, const r64 depth_threshold) const;
    std::vector<KeyFrame::Ptr> get_frames() const;
private:

    void detect_feature(std::vector<std::vector<Feature::Ptr>>& feat_pts,
                        std::vector<MatchAdjacent>& matching);

    void estimation_motion(const std::vector<MatchAdjacent>& matching,
                           std::vector<std::vector<Feature::Ptr>>& feat_pts);
private:
    std::vector<KeyFrame::Ptr> frames;
    VisibilityGraph vis_graph;
    std::vector<Landmark::Ptr> landmarks;
    Camera::Ptr camera;
};


};

#endif // 