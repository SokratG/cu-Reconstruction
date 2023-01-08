#include "feature_matcher.hpp"
#include "cr_exception.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <glog/logging.h>

namespace curec {


MatchAdjacent::MatchAdjacent(const i32 _src_idx, const i32 _dst_idx, 
                             const std::vector<cv::DMatch>& _match) :
                             src_idx(_src_idx), dst_idx(_dst_idx), 
                             match(std::make_shared<std::vector<cv::DMatch>>(_match)) {
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
    std::vector<MatchAdjacent> matching;
    for (i32 i = 0; i < adj_size; ++i) {
        for (i32 j = i + 1; j < adj_size; ++j) {
            std::vector<cv::DMatch> match;
            match_descriptors(matcher, descriptors[i], descriptors[j], mcfg, match); 
            MatchAdjacent match_adj(i, j, match);
            matching.emplace_back(match_adj);
        }
    }

    return matching;
}


std::vector<MatchAdjacent> ransac_filter_outlier(const std::vector<MatchAdjacent>& matches,
                                                 const std::vector<std::vector<Feature>>& feat_pts,
                                                 const cv::Mat intrinsic,
                                                 const r64 prob,
                                                 const r64 threshold,
                                                 const i32 min_inlier) {
    std::vector<MatchAdjacent> matching;
    const i32 adj_size = matches.size();

    for (auto ma_it = matches.begin(); ma_it != matches.end(); ma_it++) {
        const i32 src_idx = ma_it->src_idx;
        const i32 dst_idx = ma_it->dst_idx;
        const i32 match_size = ma_it->match->size();
        
        std::vector<cv::Point2d> src(match_size), dst(match_size);
        for (i32 i = 0; i < match_size; ++i) {
            src[i] = feat_pts[src_idx][ma_it->match->at(i).queryIdx].position.pt;
            dst[i] = feat_pts[dst_idx][ma_it->match->at(i).trainIdx].position.pt;
        }
        
        std::vector<cv::DMatch> match = *(ma_it->match);

        cv::Mat inlier_mask; // mask inlier points -> "status": 0 - outlier, 1 - inlier
        cv::findEssentialMat(src, dst, intrinsic, cv::FM_RANSAC, 0.9, 3.5, inlier_mask);
        const auto inliers_size = cv::countNonZero(inlier_mask) ;
        if (inliers_size < min_inlier) {
            LOG(WARNING) << "images " << src_idx << " and " << dst_idx << " don't have enough inliers: " << inliers_size; 
            continue;
        }

        std::vector<cv::DMatch> inliers(inliers_size);
        for (i32 i = 0, step = 0; i < inlier_mask.rows; ++i) {
            if (inlier_mask.at<byte>(i)) {
                inliers[step] = ma_it->match->at(i);
                step += 1;
            } 
        }
        
        MatchAdjacent match_adj(src_idx, dst_idx, inliers);
        matching.emplace_back(match_adj);
    }

    return matching;
}


};