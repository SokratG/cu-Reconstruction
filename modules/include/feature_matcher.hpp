#ifndef CUREC_LIB_FEATURE_MATCHER_HPP
#define CUREC_LIB_FEATURE_MATCHER_HPP

#include "types.hpp"
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


struct MatcherConfig {
    r32 brute_max_dist = 10000;
    r32 brute_min_dist = 0;
    r32 ratio_threshold = 0.75;
    r32 min_conf_dist = 30;
    i32 k_nn = 2;
};


struct MatchAdjacent {
    i32 src_idx;
    i32 dst_idx;
    std::shared_ptr<std::vector<cv::DMatch>> match;
    MatchAdjacent() = delete;
    MatchAdjacent(const i32 src_idx, const i32 dst_idx, const std::vector<cv::DMatch>& match);
};

std::vector<MatchAdjacent> feature_matching(const FeatureMatcherBackend backend,
                                            const std::vector<cv::Mat>& descriptors,
                                            const MatcherConfig& mcfg);

};


#endif