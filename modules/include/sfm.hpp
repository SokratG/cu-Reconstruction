#ifndef CUREC_LIB_SFM_HPP
#define CUREC_LIB_SFM_HPP

#include "keyframe.hpp"
#include "feature.hpp"
#include "camera.hpp"
#include "landmark.hpp"
#include <string_view>
#include <vector>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/core.hpp>


namespace curec {


struct MatchAdjacent;

class Sfm {
public:
    Sfm(const Camera::Ptr camera);
    bool add_frame(const cv::cuda::GpuMat frame);
    void run_pipeline();
    void store_to_ply(const std::string_view& ply_filepath, const r64 depth_threshold = 40.0);
private:

    void detect_feature(std::vector<std::vector<Feature::Ptr>>& feat_pts,
                        std::vector<MatchAdjacent>& matching);

    void estimation_motion(const std::vector<MatchAdjacent>& matching,
                           std::vector<std::vector<Feature::Ptr>>& feat_pts);

    VisibilityGraph build_landmarks_graph(const std::vector<MatchAdjacent>& ma,
                                          std::vector<std::vector<Feature::Ptr>>& feat_pts);
private:
    std::vector<KeyFrame::Ptr> frames;
    VisibilityGraph landmarks;
    Camera::Ptr camera;
};


};

#endif // 