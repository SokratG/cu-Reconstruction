#include "feature_matcher.hpp"
#include "cr_exception.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <glog/logging.h>

namespace curec {

MatchAdjacent::MatchAdjacent(const i32 _src_idx) : src_idx(_src_idx), dst_idx(-1) {

}

void match_descriptors(cv::Ptr<cv::DescriptorMatcher> matcher,
                       const cv::Mat& descriptors_src,
                       const cv::Mat& descriptors_dst,
                       const MatcherConfig& mcfg,
                       std::vector<cv::DMatch>& matches) {
    std::vector<std::vector<cv::DMatch>> knn_match;
    matcher->knnMatch(descriptors_src, descriptors_dst, knn_match, mcfg.k_nn);
    const r32 ratio_thresh = mcfg.ratio_threshold;
    for (i32 i = 0; i < knn_match.size(); i++) {
        if (knn_match[i][0].distance < ratio_thresh * knn_match[i][1].distance) {
            matches.emplace_back(knn_match[i][0]);
        }
    } 
}

std::vector<MatchAdjacent> feature_matching(const FeatureMatcherBackend backend,
                                            const std::vector<cv::Mat>& descriptors,
                                            const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                            const Mat3& intrinsic,
                                            const MatcherConfig& mcfg) {
    cv::Ptr<cv::DescriptorMatcher> matcher;                                            
    switch (backend) {
        case FeatureMatcherBackend::BRUTEFORCE:
            matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_SL2);
            break;
        case FeatureMatcherBackend::FLANN:
            matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
            break;
        default:
            throw CuRecException("The given feature matcher backend is not allowed!");
    }
    
    const i32 adj_size = descriptors.size();
    cv::Mat K;
    cv::eigen2cv(intrinsic, K);
    
    std::vector<MatchAdjacent> matching;
    for (i32 i = 0; i < adj_size; ++i) {
        MatchAdjacent match_adj(i);
        auto max_inliers = 0;
        for (i32 j = i + 1; j < adj_size; ++j) {
            std::vector<cv::DMatch> match;
            match_descriptors(matcher, descriptors[i], descriptors[j], mcfg, match);
            const auto match_size = match.size();

            std::vector<cv::Point2d> src(match_size), dst(match_size);
            for (i32 k = 0; k < match_size; ++k) {
                src[k] = feat_pts[i].at(match.at(k).queryIdx)->position.pt;
                dst[k] = feat_pts[j].at(match.at(k).trainIdx)->position.pt;
            }

            cv::Mat inlier_mask; // mask inlier points -> "status": 0 - outlier, 1 - inlier
            cv::findEssentialMat(src, dst, K, cv::FM_RANSAC, mcfg.prob, mcfg.threshold, inlier_mask);
            const auto inliers_size = cv::countNonZero(inlier_mask) ;
            if (inliers_size < mcfg.min_inlier) {
                LOG(WARNING) << "images " << i << " and " << j << " don't have enough inliers: " << inliers_size; 
                continue;
            }

            if (max_inliers < inliers_size) {
                match_adj.dst_idx = j;
                match_adj.match = match;
                max_inliers = inliers_size;
            }           
        }
        matching.emplace_back(match_adj);
    }

    return matching;
}


};