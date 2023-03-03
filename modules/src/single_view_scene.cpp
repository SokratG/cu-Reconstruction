#include "single_view_scene.hpp"
#include "feature_detector.hpp"

#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>


namespace cuphoto {


SingleViewScene::SingleViewScene(const Camera::Ptr _camera) : camera(_camera) {

}

cudaPointCloud::Ptr SingleViewScene::get_point_cloud() const {
    return cuda_pc;
}

bool SingleViewScene::store_to_ply(const std::string& ply_filepath) const {
    return cuda_pc->save_ply(ply_filepath);
}


void SingleViewScene::estimate_pose(KeyFrame::Ptr frame_rgb, KeyFrame::Ptr frame_depth, const Config& cfg) {
    FeatureDetectorBackend backend = static_cast<FeatureDetectorBackend>(cfg.get<i32>("feature.type", 1));
    FeatureDetector fd(backend, cfg);
    std::vector<Feature::Ptr> features;
    cv::cuda::GpuMat descriptor;
    fd.detectAndCompute(frame_rgb, features, descriptor);
    
    std::vector<cv::Point2f> img_pts;
    for (const auto& feature : features) {
        img_pts.emplace_back(feature->position.pt);
    }

    cv::Mat depth;
    frame_depth->frame().download(depth);

    Mat3 K = camera->K();
    std::vector<cv::Point3f> obj_pts;
    for (const auto& feature : features) {
        cv::Point2f src_pt(feature->position.pt.x, feature->position.pt.y);
        const r32 z = depth.at<r32>(src_pt);
        if (z == 0.0)
            continue;
        const r32 x = (src_pt.x - K(0, 2)) * z / K(0, 0);
        const r32 y = (src_pt.y - K(1, 2)) * z / K(1, 1);
        obj_pts.emplace_back(cv::Point3f(x, y, z));
    }

    cv::Mat rvec(3, 1, cv::DataType<r64>::type);
    cv::Mat tvec(3, 1, cv::DataType<r64>::type);

    cv::Mat camera_matrix;
    cv::eigen2cv(K, camera_matrix);

    cv::solvePnPRansac(obj_pts, img_pts, camera_matrix, cv::noArray(), rvec, tvec);

    cv::solvePnPRefineLM(obj_pts, img_pts, camera_matrix, cv::noArray(), rvec, tvec);

    cv::Mat cvR;
    cv::Rodrigues(rvec, cvR);
    Mat3 R;
    cv::cv2eigen(cvR, R);

    Vec3 t;
    cv::cv2eigen(tvec, t);

    frame_rgb->pose(R, t);
    frame_depth->pose(R, t);
}



};