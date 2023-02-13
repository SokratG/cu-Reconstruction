#include "ceres_optim.hpp"
#include "keyframe.hpp"
#include "cp_exception.hpp"
#include <glog/logging.h>


namespace cuphoto {

CeresCameraModel::CeresCameraModel(const SE3& camera_pose) {
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
CeresOptimizerReprojection::CeresOptimizerReprojection(const TypeReprojectionError tre, const r64 _loss_width) : 
                                                       type_err(tre), loss_width(_loss_width) {
    optim_problem = std::make_shared<ceres::Problem>();
}

ceres::CostFunction* CeresOptimizerReprojection::get_cost_function(const Mat3& K, const Vec3& world_pt, const Vec2& obs_pt) {
    switch (type_err) {
        case TypeReprojectionError::REPROJECTION_POSE_POINT:
            return ReprojectionErrorPosePoint::create(K, obs_pt);
        case TypeReprojectionError::REPROJECTION_POSE:
            return ReprojectionErrorPose::create(K, world_pt, obs_pt);
        default:
            throw CuPhotoException("Error: Unknown type of reprojection error!");
    }
}

void CeresOptimizerReprojection::reset() {
    optim_problem = std::make_shared<ceres::Problem>();
    ceres_cameras = std::unordered_map<ui64, CeresCameraModel>();
    ceres_obseravations = std::unordered_map<ui64, CeresObservation>();
}


void CeresOptimizerReprojection::build_blocks_reprojection(const VisibilityGraph& vis_graph,
                                                           const std::vector<Landmark::Ptr>& landmarks, 
                                                           const std::vector<KeyFrame::Ptr>& frames,
                                                           const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                                           const Camera::Ptr camera) {
    if (landmarks.empty() || frames.empty())
        throw CuPhotoException("Error: the given(landmarks | frames) data is empty!");

    const Mat3 K = camera->K();
    for (auto cam_idx = 0; cam_idx < frames.size(); ++cam_idx) {
        CeresCameraModel c_cam(frames[cam_idx]->pose());
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

void CeresOptimizerReprojection::add_block(CeresCameraModel& ceres_camera, 
                                           CeresObservation& landmark, 
                                           const Vec2& observ_pt,
                                           const Mat3& K) {
    ceres::CostFunction* cost_f = get_cost_function(K, landmark.position(), observ_pt);
    ceres::LossFunction* loss_f = loss_width > 0 ? new ceres::CauchyLoss(6.5) : nullptr;
    switch (type_err) {
        case TypeReprojectionError::REPROJECTION_POSE:
            optim_problem->AddResidualBlock(cost_f, loss_f, ceres_camera.raw_camera_param);
            break;
        default:
            optim_problem->AddResidualBlock(cost_f, loss_f, ceres_camera.raw_camera_param, landmark.obs);
            break;
    }
    
}

void CeresOptimizerReprojection::optimize(const i32 n_iteration, const i32 num_threads, const bool fullreport) {
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


void CeresOptimizerReprojection::store_result(std::vector<Landmark::Ptr>& landmarks, 
                                              std::vector<KeyFrame::Ptr>& frames) 
{
    for (auto cam_idx = 0; cam_idx < frames.size(); ++cam_idx) {
        frames.at(cam_idx)->pose(ceres_cameras.at(cam_idx).pose());
    }

    for (auto lm_idx = 0; lm_idx < landmarks.size(); ++lm_idx) {
        landmarks.at(lm_idx)->pose(ceres_obseravations.at(lm_idx).position());
    }
}

void CeresOptimizerReprojection::build_blocks_icp(const std::unordered_map<i32, ConnectionPoints>& pts3D,
                                 const std::vector<MatchAdjacent>& ma,
                                 const std::vector<KeyFrame::Ptr>& frames) {
    throw CuPhotoException("CeresOptimizerReprojection::build_blocks_icp(): Not implemented yet!");
}

void CeresOptimizerReprojection::store_result(std::vector<KeyFrame::Ptr>& frames) {
    throw CuPhotoException("CeresOptimizerReprojection::store_result(): Not implemented yet!");
}


// ===================================================================================
CeresOptimizerICP::CeresOptimizerICP(const r64 _loss_width) : loss_width(_loss_width) {
    optim_problem = std::make_shared<ceres::Problem>();
}

void CeresOptimizerICP::build_blocks_icp(const std::unordered_map<i32, ConnectionPoints>& pts3D,
                                         const std::vector<MatchAdjacent>& ma,
                                         const std::vector<KeyFrame::Ptr>& frames) {
    if (pts3D.empty() || frames.empty())
        throw CuPhotoException("Error: the given(pts3D | frames) data is empty!");
    
    const Mat3 K = Mat3::Identity();
    for (auto cam_idx = 0; cam_idx < frames.size(); ++cam_idx) {
        CeresCameraModel c_cam(frames[cam_idx]->pose());
        ceres_cameras.insert({cam_idx, c_cam});
    }

    for (const auto& adj : ma) {
        const i32 src_idx = adj.src_idx;
        if (adj.dst_idx < 0)
            continue;
        const i32 frame_id = adj.dst_idx;
        const auto& pts = pts3D.at(src_idx);
        for (const auto& pair_pt : pts) {
            const auto src_pt = pair_pt.first;
            const auto dst_pt = pair_pt.second;
            add_block(ceres_cameras.at(frame_id), src_pt, dst_pt);
        }
    }

}

void CeresOptimizerICP::reset() {
    optim_problem = std::make_shared<ceres::Problem>();
    ceres_cameras = std::unordered_map<ui64, CeresCameraModel>();
}


void CeresOptimizerICP::add_block(CeresCameraModel& ceres_camera, const Vec3& src_pt, const Vec3& dst_pt) {
    ceres::CostFunction* cost_f = ErrorICPPose::create(src_pt, dst_pt);
    ceres::LossFunction* loss_f = loss_width > 0 ? new ceres::CauchyLoss(loss_width) : nullptr;
    optim_problem->AddResidualBlock(cost_f, loss_f, ceres_camera.raw_camera_param);
}


void CeresOptimizerICP::optimize(const i32 n_iteration, const i32 num_threads, const bool fullreport) {
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

void CeresOptimizerICP::store_result(std::vector<KeyFrame::Ptr>& frames) {
    for (auto cam_idx = 0; cam_idx < frames.size(); ++cam_idx) {
        frames.at(cam_idx)->pose(ceres_cameras.at(cam_idx).pose());
    }
}


void CeresOptimizerICP::build_blocks_reprojection(const VisibilityGraph& vis_graph,
                                          const std::vector<Landmark::Ptr>& landmarks,
                                          const std::vector<KeyFrame::Ptr>& frames,
                                          const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                          const Camera::Ptr camera) {
    throw CuPhotoException("CeresOptimizerICP::build_blocks_reprojection(): Not implemented yet!");
}

void CeresOptimizerICP::store_result(std::vector<Landmark::Ptr>& landmarks,
                             std::vector<KeyFrame::Ptr>& frames) {
    throw CuPhotoException("CeresOptimizerICP::store_result(): Not implemented yet!");
}

};
