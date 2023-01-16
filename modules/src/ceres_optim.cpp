#include "ceres_optim.hpp"
#include "keyframe.hpp"
#include "cr_exception.hpp"
#include <glog/logging.h>

namespace curec {

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
    raw_camera_param[8] = K(1, 1);
    camera_center = Vec2(K(0, 2), K(2, 2));
}

Mat3 CeresCameraModel::K() const {
    Mat3 K;
    K << raw_camera_param[7], 0, camera_center.x(), 0, raw_camera_param[8], camera_center.y(), 0, 0, 1;
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
            return ReprojectionErrorRt::create(K, pt);
        case TypeReprojectionError::REPROJECTION_FOCAL_RT:
            return ReprojectionErrorFocalRt::create(Vec2(K(0, 2), K(1, 2)), pt);
        default:
            throw CuRecException("Error: Unknown type of reprojection error!");
    }
}

void CeresOptimizer::reset() {
    optim_problem = std::make_shared<ceres::Problem>();
    ceres_cameras = std::unordered_map<uuid, CeresCameraModel>();
    ceres_obseravations = std::unordered_map<uuid, CeresObservation>();
}

void CeresOptimizer::build_blocks(const VisibilityGraph& landmarks,
                                  const std::vector<KeyFrame::Ptr>& frames,
                                  const Camera::Ptr camera) {
    if (landmarks.empty() || frames.empty())
        throw CuRecException("Error: the given(landmarks | frames) data is empty!");

    const Mat3 K = camera->K();
    for (const auto frame : frames) {
        CeresCameraModel c_cam(frame->pose(), K);
        ceres_cameras.insert({frame->id, c_cam});
    }
    
    for (const auto landmark_pair : landmarks) {
        const auto landmark = landmark_pair.second;
        const auto frame_id = landmark->observation()->frame.lock()->id;
        const auto& ccam = ceres_cameras.at(frame_id);
        const auto key_pt = landmark->observation()->position.pt;
        const Vec2 obs_pt(key_pt.x, key_pt.y);
        CeresObservation co(landmark->pose());
        ceres_obseravations.insert({landmark->id, co});
        add_block(ceres_cameras.at(frame_id), ceres_obseravations.at(landmark->id), obs_pt, K);
        
    }
}

void CeresOptimizer::add_block(CeresCameraModel& ceres_camera, 
                               CeresObservation& landmark, 
                               const Vec2& observ_pt,
                               const Mat3& K) {
    ceres::CostFunction* cost_f = get_cost_function(K, observ_pt);
    ceres::LossFunction* loss_f = loss_width > 0 ? new ceres::HuberLoss(loss_width) : nullptr;
    optim_problem->AddResidualBlock(cost_f, loss_f, ceres_camera.raw_camera_param, landmark.obs);
}

void CeresOptimizer::optimize(const i32 n_iteration, const i32 num_threads, const bool fullreport) {
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR; // ITERATIVE_SCHUR
    options.num_threads = num_threads;
    if (n_iteration > 0) 
        options.max_num_iterations = n_iteration;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, optim_problem.get(), &summary);

    if (fullreport)
        LOG(INFO) << summary.FullReport();
}


void CeresOptimizer::store_result(VisibilityGraph& landmarks, std::vector<KeyFrame::Ptr>& frames) {
    for (auto& frame : frames) {
        const auto& c_cam = ceres_cameras.at(frame->id);
        frame->pose(c_cam.pose());
    }

    for (auto& landmark : landmarks) {
        const auto& c_obs = ceres_obseravations.at(landmark.second->id);
        landmark.second->pose(c_obs.position());
    }
}


};
