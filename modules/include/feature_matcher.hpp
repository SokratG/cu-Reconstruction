#ifndef CUREC_LIB_FEATURE_MATCHER_HPP
#define CUREC_LIB_FEATURE_MATCHER_HPP

#include "types.hpp"
#include "feature.hpp"
#include <string>
#include <vector>
#include <map>
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

struct OutlierMatcherConfig {
    r64 prob = 0.9;
    r64 threshold = 2.5;
    i32 min_inlier = 50;
};



struct MatchAdjacent {
    i32 src_idx;
    std::map<i32, std::vector<cv::DMatch>> ord_match;
    MatchAdjacent() = delete;
    MatchAdjacent(const i32 _src_idx);
};

std::vector<MatchAdjacent> feature_matching(const FeatureMatcherBackend backend,
                                            const std::vector<cv::Mat>& descriptors,
                                            const MatcherConfig& mcfg = MatcherConfig());

std::vector<MatchAdjacent> ransac_filter_outlier(const std::vector<MatchAdjacent>& matches,
                                                 const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                                 const Mat3& intrinsic,
                                                 const OutlierMatcherConfig& cfg = OutlierMatcherConfig());

};

#endif