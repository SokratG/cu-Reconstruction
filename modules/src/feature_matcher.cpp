#include "feature_matcher.hpp"
#include "cp_exception.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <glog/logging.h>

namespace cuphoto {


MatchAdjacent::MatchAdjacent(const i32 _src_idx) : src_idx(_src_idx), dst_idx(-1) {

}

void Matcher::filter(const std::vector<std::vector<cv::DMatch>>& knn_match,
                     const MatcherConfig& mcfg, std::vector<cv::DMatch>& matches) {
    const r32 ratio_thresh = mcfg.ratio_threshold;
    for (i32 i = 0; i < knn_match.size(); i++) {
        if (knn_match[i][0].distance < ratio_thresh * knn_match[i][1].distance) {
            matches.emplace_back(knn_match[i][0]);
        }
    }
}


class FLANNMatcher : public Matcher {
public:
    FLANNMatcher() {
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }
    void match(cv::InputArray descriptor_src, cv::InputArray descriptor_dst, 
               const MatcherConfig& mcfg, std::vector<cv::DMatch>& matches) override;
private:
    cv::Ptr<cv::DescriptorMatcher> matcher;
};


class CudaBruteForceMatcher : public Matcher {
public:
    CudaBruteForceMatcher() {
        matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
    }
    void match(cv::InputArray descriptor_src, cv::InputArray descriptor_dst, 
               const MatcherConfig& mcfg, std::vector<cv::DMatch>& matches) override;
private:
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher;
};


void FLANNMatcher::match(cv::InputArray descriptor_src, cv::InputArray descriptor_dst, 
                         const MatcherConfig& mcfg, std::vector<cv::DMatch>& matches) {
    std::vector<std::vector<cv::DMatch>> knn_match;
    cv::Mat d_src(descriptor_src.getGpuMat()), d_dst(descriptor_dst.getGpuMat());
    matcher->knnMatch(d_src, d_dst, knn_match, mcfg.k_nn);
    filter(knn_match, mcfg, matches);
}

void CudaBruteForceMatcher::match(cv::InputArray descriptor_src, cv::InputArray descriptor_dst, 
                                  const MatcherConfig& mcfg, std::vector<cv::DMatch>& matches) {
    std::vector<std::vector<cv::DMatch>> knn_match;                     
    matcher->knnMatch(descriptor_src, descriptor_dst, knn_match, mcfg.k_nn);
    filter(knn_match, mcfg, matches);                   
}


cv::Ptr<Matcher> create_matcher(const MatcherConfig& mcfg) {
    switch (mcfg.backend) {
        case FeatureMatcherBackend::BRUTEFORCE:
            return cv::makePtr<CudaBruteForceMatcher>();
        case FeatureMatcherBackend::FLANN:
            return cv::makePtr<FLANNMatcher>();
        default:
            throw CuPhotoException("The given feature matcher backend is not allowed!");
    }
}


std::vector<MatchAdjacent> feature_matching(const std::vector<cv::cuda::GpuMat>& descriptors,
                                            const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                            const Mat3& intrinsic,
                                            const MatcherConfig& mcfg) {
    cv::Ptr<Matcher> matcher = create_matcher(mcfg);                                            
    
    const i32 adj_size = descriptors.size();
    cv::Mat K;
    cv::eigen2cv(intrinsic, K);
    
    std::vector<MatchAdjacent> matching;
    for (i32 i = 0; i < adj_size; ++i) {
        MatchAdjacent match_adj(i);
        auto max_inliers = 0;
        for (i32 j = i + 1; j < adj_size; ++j) {
            std::vector<cv::DMatch> match;
            matcher->match(descriptors[i], descriptors[j], mcfg, match);
            const auto match_size = match.size();

            std::vector<cv::Point2d> src(match_size), dst(match_size);
            for (i32 k = 0; k < match_size; ++k) {
                src[k] = feat_pts[i].at(match.at(k).queryIdx)->position.pt;
                dst[k] = feat_pts[j].at(match.at(k).trainIdx)->position.pt;
            }

            cv::Mat inlier_mask; // mask inlier points -> "status": 0 - outlier, 1 - inlier
            cv::findEssentialMat(src, dst, K, cv::FM_RANSAC, mcfg.prob, mcfg.threshold, inlier_mask);
            auto inliers_size = 0;
            for (i32 k = 0; k < inlier_mask.rows; ++k) {
                if (inlier_mask.at<byte>(k)) {
                    inliers_size += 1;
                } else {
                    feat_pts[i].at(match.at(k).queryIdx)->outlier(true);
                    feat_pts[j].at(match.at(k).trainIdx)->outlier(true);
                }
            }

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