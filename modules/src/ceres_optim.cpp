#include "ceres_optim.hpp"
#include "keyframe.hpp"
#include "cp_exception.hpp"
#include <glog/logging.h>


namespace cuphoto {

CeresCameraModel::CeresCameraModel(const SE3& camera_pose, const Mat3& K) {
    Mat3 R = camera_pose.rotationMatrix();
    Vec3 t = camera_pose.translation();
    Eigen::Quaterniond q(R);
    raw_camera_param[0] = q.w();
    raw_camera_param[1] = q.x();
    raw_camera_param[2] = q.y();
    raw_camera_param[3] = q.z();
    raw_camera_param[4] = t.x();
    raw_camera_param[5] = t.y();
    raw_camera_param[6] = t.z();
    raw_camera_param[7] = K(0, 0);
    camera_center = Vec2(K(0, 2), K(1, 2));
}

Mat3 CeresCameraModel::K() const {
    Mat3 K;
    K << raw_camera_param[7], 0, camera_center.x(), 0, raw_camera_param[7], camera_center.y(), 0, 0, 1;
    return K;
}

SE3 CeresCameraModel::pose() const {
    Eigen::Quaterniond q(raw_camera_param[0], raw_camera_param[1], raw_camera_param[2], raw_camera_param[3]);
    Vec3 t(raw_camera_param[4], raw_camera_param[5], raw_camera_param[6]);
    SE3 cam_pose(q, t);
    return cam_pose;
}

CeresObservation::CeresObservation(const Vec3& pt3d) {
    obs[0] = pt3d.x();
    obs[1] = pt3d.y();
    obs[2] = pt3d.z();
}

Vec3 CeresObservation::position() const {
    Vec3 pos(obs[0], obs[1], obs[2]);
    return pos;
}

// ===================================================================================

CeresOptimizer::CeresOptimizer(const TypeReprojectionError tre, const r64 _loss_width) : 
                               type_err(tre), loss_width(_loss_width) {
    optim_problem = std::make_shared<ceres::Problem>();
}

ceres::CostFunction* CeresOptimizer::get_cost_function(const Mat3& K, const Vec2& pt) {
    switch (type_err) 
    {
        case TypeReprojectionError::REPROJECTION_RT:
            return ReprojectionErrorPose::create(K, pt);
        case TypeReprojectionError::REPROJECTION_FOCAL_RT:
            return ReprojectionErrorFocalRt::create(Vec2(K(0, 2), K(1, 2)), pt);
        default:
            throw CuPhotoException("Error: Unknown type of reprojection error!");
    }
}

void CeresOptimizer::reset() {
    optim_problem = std::make_shared<ceres::Problem>();
    ceres_cameras = std::unordered_map<ui64, CeresCameraModel>();
    ceres_obseravations = std::unordered_map<ui64, CeresObservation>();
}

void CeresOptimizer::build_blocks(const VisibilityGraph& vis_graph,
                                  const std::vector<Landmark::Ptr>& landmarks, 
                                  const std::vector<KeyFrame::Ptr>& frames,
                                  const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                  const Camera::Ptr camera) {
    if (landmarks.empty() || frames.empty())
        throw CuPhotoException("Error: the given(landmarks | frames) data is empty!");

    const Mat3 K = camera->K();
    for (auto cam_idx = 0; cam_idx < frames.size(); ++cam_idx) {
        CeresCameraModel c_cam(frames[cam_idx]->pose(), K);
        ceres_cameras.insert({cam_idx, c_cam});
    }
    
    for (const auto vis_node : vis_graph) {
        const auto vnode = vis_node.second;
        const auto frame_id = vnode->frame_idx();
        const auto pt_idx = vnode->obs_idx();
        const auto key_pt = feat_pts.at(frame_id).at(pt_idx)->position.pt;
        const Vec2 obs_pt(key_pt.x, key_pt.y);
        const auto lm_idx = vnode->landmark_idx();
        CeresObservation co(landmarks.at(lm_idx)->pose());
        ceres_obseravations.insert({lm_idx, co});
        add_block(ceres_cameras.at(frame_id), ceres_obseravations.at(lm_idx), obs_pt, K);
    }
}

void CeresOptimizer::add_block(CeresCameraModel& ceres_camera, 
                               CeresObservation& landmark, 
                               const Vec2& observ_pt,
                               const Mat3& K) {
    ceres::CostFunction* cost_f = get_cost_function(K, observ_pt);
    ceres::LossFunction* loss_f = loss_width > 0 ? new ceres::CauchyLoss(6.5) : nullptr;
    optim_problem->AddResidualBlock(cost_f, loss_f, ceres_camera.raw_camera_param, landmark.obs);
}

void CeresOptimizer::optimize(const i32 n_iteration, const i32 num_threads, const bool fullreport) {
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR; // ITERATIVE_SCHUR SPARSE_SCHUR
    options.num_threads = num_threads;
    if (n_iteration > 0) 
        options.max_num_iterations = n_iteration;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, optim_problem.get(), &summary);

    if (fullreport)
        LOG(INFO) << summary.FullReport();
}


void CeresOptimizer::store_result(std::vector<Landmark::Ptr>& landmarks, 
                                  std::vector<KeyFrame::Ptr>& frames) 
{
    for (auto cam_idx = 0; cam_idx < frames.size(); ++cam_idx) {
        frames.at(cam_idx)->pose(ceres_cameras.at(cam_idx).pose());
    }

    for (auto lm_idx = 0; lm_idx < landmarks.size(); ++lm_idx) {
        landmarks.at(lm_idx)->pose(ceres_obseravations.at(lm_idx).position());
    }
}


};
