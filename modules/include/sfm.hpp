#ifndef CUPHOTO_LIB_SFM_HPP
#define CUPHOTO_LIB_SFM_HPP

#include "config.hpp"
#include "visibility_graph.hpp"
#include <vector>
#include <limits>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/core.hpp>


namespace cuphoto {

constexpr r32 low_threshold = -std::numeric_limits<r32>::max();
constexpr r32 high_threshold = std::numeric_limits<r32>::max();

struct MatchAdjacent;

class Sfm {
public:
    Sfm(const Camera::Ptr camera);
    bool add_frame(const cv::cuda::GpuMat frame);
    void run_pipeline(const Config& cfg);
    void store_to_ply(const std::string& ply_filepath, 
                      const r32 x_min = low_threshold, const r32 x_max = high_threshold,
                      const r32 y_min = low_threshold, const r32 y_max = high_threshold,
                      const r32 depth = high_threshold) const;
    std::vector<KeyFrame::Ptr> get_frames() const;
private:

    void filter_outlier_frames(const Config& cfg,
                               std::vector<std::vector<Feature::Ptr>>& feat_pts,
                               std::vector<cv::cuda::GpuMat>& descriptors);

    void detect_feature(const Config& cfg,
                        std::vector<std::vector<Feature::Ptr>>& feat_pts,
                        std::vector<cv::cuda::GpuMat>& descriptors);
    
    void matching_feature(const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                          const std::vector<cv::cuda::GpuMat>& descriptors,
                          const Config& cfg,
                          std::vector<MatchAdjacent>& matching);

    void estimation_motion(const std::vector<MatchAdjacent>& matching,
                           const Config& cfg,
                           std::vector<std::vector<Feature::Ptr>>& feat_pts);
private:
    std::vector<KeyFrame::Ptr> frames;
    VisibilityGraph vis_graph;
    std::vector<Landmark::Ptr> landmarks;
    Camera::Ptr camera;
};


};

#endif // 