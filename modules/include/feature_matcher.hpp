#ifndef CUREC_LIB_FEATURE_MATCHER_HPP
#define CUREC_LIB_FEATURE_MATCHER_HPP

#include "types.hpp"
#include "feature.hpp"
#include "feature_detector.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/cuda.hpp>

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
    r64 prob = 0.9;
    r64 threshold = 2.5;
    i32 min_inlier = 50;
    FeatureMatcherBackend backend = FeatureMatcherBackend::BRUTEFORCE;
};

struct OutlierMatcherConfig {
    r64 prob = 0.9;
    r64 threshold = 2.5;
    i32 min_inlier = 50;
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
                       const MatcherConfig& mcfg, std::vector<cv::DMatch>& matches) = 0;
protected:
    virtual void filter(const std::vector<std::vector<cv::DMatch>>& knn_match,
                        const MatcherConfig& mcfg, std::vector<cv::DMatch>& matches);
};


std::vector<MatchAdjacent> feature_matching(const std::vector<cv::cuda::GpuMat>& descriptors,
                                            const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                            const Mat3& intrinsic,
                                            const MatcherConfig& mcfg = MatcherConfig());

};

#endif