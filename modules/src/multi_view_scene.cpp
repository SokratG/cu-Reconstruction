#include "multi_view_scene.hpp"

namespace cuphoto {

MultiViewScene::MultiViewScene(const Camera::Ptr _camera) : camera(_camera) {

}

void MultiViewScene::detect_feature(const std::vector<KeyFrame::Ptr>& frames,
                                    std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                    std::vector<cv::cuda::GpuMat>& descriptors) 
{
    FeatureDetector fd(FeatureDetectorBackend::SIFT, "");
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
                                      std::vector<MatchAdjacent>& matching) 
{
    matching = feature_matching(descriptors,
                                feat_pts,
                                camera->K());
}

cudaPointCloud::Ptr MultiViewScene::get_point_cloud() const {
    return cuda_pc;
}

bool MultiViewScene::store_to_ply(const std::string& ply_filepath) const {
    return cuda_pc->save_ply(ply_filepath);
}


};

