#ifndef CUREC_LIB_FEATURE_MATCHER_HPP
#define CUREC_LIB_FEATURE_MATCHER_HPP

#include "types.hpp"
#include "feature.hpp"
#include <string>
#include <vector>
#include <memory>
#include <opencv4/opencv2/features2d.hpp>

namespace curec {

enum class FeatureMatcherBackend {
    BRUTEFORCE,
    FLANN,
    UNKNOWN
};

// TODO: add config from file
struct MatcherConfig {
    r32 ratio_threshold = 0.8;
    i32 k_nn = 2;
};


struct MatchAdjacent {
    i32 src_idx;
    i32 dst_idx;
    std::vector<cv::DMatch> match;
    MatchAdjacent() = delete;
    MatchAdjacent(const i32 src_idx, const i32 dst_idx, const std::vector<cv::DMatch>& match);
};

std::vector<MatchAdjacent> feature_matching(const FeatureMatcherBackend backend,
                                            const std::vector<cv::Mat>& descriptors,
                                            const MatcherConfig& mcfg = MatcherConfig());

std::vector<MatchAdjacent> ransac_filter_outlier(const std::vector<MatchAdjacent>& matches,
                                                const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                                const cv::Mat intrinsic,
                                                const r64 prob = 0.9,
                                                const r64 threshold = 3.5,
                                                const i32 min_inlier = 50);

};

#endif