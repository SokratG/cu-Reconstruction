#include "multi_view_scene.hpp"
#include "feature_detector.hpp"

namespace cuphoto {

MultiViewScene::MultiViewScene(const Camera::Ptr _camera) : camera(_camera) {

}

void MultiViewScene::detect_feature(const std::vector<KeyFrame::Ptr>& frames,
                                    const Config& cfg,
                                    std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                    std::vector<cv::cuda::GpuMat>& descriptors) 
{
    FeatureDetectorBackend backend = static_cast<FeatureDetectorBackend>(cfg.get<i32>("feature.type", 1));
    FeatureDetector fd(backend, cfg);
    for (const auto frame : frames) {
        std::vector<Feature::Ptr> kpts;
        cv::cuda::GpuMat descriptor;
        fd.detectAndCompute(frame, kpts, descriptor);
        descriptors.emplace_back(descriptor);
        feat_pts.emplace_back(kpts);
    }    
}


void MultiViewScene::matching_feature(const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                      const std::vector<cv::cuda::GpuMat>& descriptors,
                                      const Config& cfg,
                                      std::vector<MatchAdjacent>& matching)
{
    FeatureMatcherBackend backend = static_cast<FeatureMatcherBackend>(cfg.get<i32>("feature.matching.type", 0));
    matching = feature_matching(descriptors,
                                feat_pts,
                                camera->K(),
                                backend,
                                cfg);
}

cudaPointCloud::Ptr MultiViewScene::get_point_cloud() const {
    return cuda_pc;
}

bool MultiViewScene::store_to_ply(const std::string& ply_filepath) const {
    return cuda_pc->save_ply(ply_filepath);
}


};

