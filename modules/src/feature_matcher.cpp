#include "feature_matcher.hpp"
#include "cr_exception.hpp"

namespace curec {


MatchAdjacent::MatchAdjacent(const i32 _src_idx, const i32 _dst_idx, 
                             const std::vector<cv::DMatch>& _match) :
                             src_idx(_src_idx), dst_idx(_dst_idx), 
                             match(std::make_shared<std::vector<cv::DMatch>>(_match)) {
}

void flann_matching(const cv::Mat& descriptors_src,
                    const cv::Mat& descriptors_dst,
                    const MatcherConfig& mcfg,
                    std::vector<cv::DMatch>& matches) {
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    cv::Mat desc_src, desc_dst;
    descriptors_src.convertTo(desc_src, CV_32F);
    descriptors_dst.convertTo(desc_dst, CV_32F);
    std::vector<std::vector<cv::DMatch>> knn_match;
    matcher->knnMatch(desc_src, desc_dst, knn_match, mcfg.k_nn);
    const r32 ratio_thresh = mcfg.ratio_threshold;
    for (i32 i = 0; i < knn_match.size(); i++) {
        if (knn_match[i][0].distance < ratio_thresh * knn_match[i][1].distance) {
            matches.emplace_back(knn_match[i][0]);
        }
    }
    
}


void bruteforce_matching(const cv::Mat& descriptors_src,
                         const cv::Mat& descriptors_dst,
                         const MatcherConfig& mcfg,
                         std::vector<cv::DMatch>& matches) {
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    std::vector<cv::DMatch> match;
    matcher->match(descriptors_src, descriptors_dst, match);
    r32 min_dist = mcfg.brute_min_dist, max_dist = mcfg.brute_max_dist;
    for (i32 i = 0; i < descriptors_src.rows; i++) {
        const r32 dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    for (i32 i = 0; i < descriptors_src.rows; i++) {
        if (match[i].distance <= std::max(2 * min_dist, mcfg.min_conf_dist)) {
            matches.emplace_back(match[i]);
        }
    }
}



std::vector<MatchAdjacent> feature_matching(const FeatureMatcherBackend backend,
                                            const std::vector<cv::Mat>& descriptors,
                                            const MatcherConfig& mcfg) {
    const i32 adj_size = descriptors.size();
    std::vector<MatchAdjacent> matching;
    for (i32 i = 0; i < adj_size; ++i) {
        for (i32 j = i + 1; j < adj_size; ++j) {
            std::vector<cv::DMatch> match;
            switch (backend) {
                case FeatureMatcherBackend::BRUTEFORCE:
                    bruteforce_matching(descriptors[i], descriptors[j], mcfg, match);
                    break;
                case FeatureMatcherBackend::FLANN:
                    flann_matching(descriptors[i], descriptors[j], mcfg, match);
                    break;
                default:
                    throw CuRecException("The given feature matcher backend is not allowed!");
            }
            MatchAdjacent match_adj(i, j, match);
            matching.emplace_back(match_adj);
        }
    }

    return matching;
}

};