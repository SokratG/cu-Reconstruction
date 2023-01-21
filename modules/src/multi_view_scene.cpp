#include "multi_view_scene.hpp"

namespace cuphoto {

MultiViewScene::MultiViewScene(const Camera::Ptr _camera) : camera(_camera) {

}

void MultiViewScene::detect_feature(const std::vector<KeyFrame::Ptr>& frames,
                                    std::vector<std::vector<Feature::Ptr>>& feat_pts, 
                                    std::vector<MatchAdjacent>& matching) 
{
    FeatureDetector fd(FeatureDetectorBackend::SIFT, "");
    std::vector<cv::cuda::GpuMat> descriptors;
    for (const auto frame : frames) {
        std::vector<Feature::Ptr> kpts;
        cv::cuda::GpuMat descriptor;
        fd.detectAndCompute(frame, kpts, descriptor);
        descriptors.emplace_back(descriptor);
        feat_pts.emplace_back(kpts);
    }

    matching = feature_matching(descriptors,
                                feat_pts,
                                camera->K());
}




};

