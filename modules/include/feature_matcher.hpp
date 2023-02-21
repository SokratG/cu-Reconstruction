#ifndef CUPHOTO_LIB_FEATURE_MATCHER_HPP
#define CUPHOTO_LIB_FEATURE_MATCHER_HPP

#include "types.hpp"
#include "config.hpp"
#include "feature.hpp"
#include <string>
#include <vector>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/cuda.hpp>

namespace cuphoto {

enum class FeatureMatcherBackend {
    BRUTEFORCE = 0,
    FLANN,
    UNKNOWN
};

struct MatchAdjacent {
    i32 src_idx;
    i32 dst_idx;
    std::vector<cv::DMatch> match;
    MatchAdjacent() = delete;
    MatchAdjacent(const i32 src_idx);
};


class Matcher {
public:
    virtual void match(cv::InputArray descriptor_src, cv::InputArray descriptor_dst, 
                       const i32 knn, const r32 ratio_threshold,
                       std::vector<cv::DMatch>& matches) = 0;
protected:
    virtual void filter(const std::vector<std::vector<cv::DMatch>>& knn_match,
                        const r32 ratio_threshold, std::vector<cv::DMatch>& matches);
};


std::vector<MatchAdjacent> feature_matching(const std::vector<cv::cuda::GpuMat>& descriptors,
                                            const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                            const Mat3& intrinsic,
                                            const FeatureMatcherBackend backend,
                                            const Config& cfg);

};

#endif