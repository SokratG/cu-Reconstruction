#include "bundle_adjustment.hpp"
#include "ceres_optim.hpp"
#include "cp_exception.hpp"

namespace cuphoto {

BundleAdjustment::BundleAdjustment(const OptimizerType opt_type,
                                   const TypeReprojectionError type_err,
                                   const Config& cfg) {
    optimizer = get_optimizer(opt_type, type_err, cfg);
}

Optimizer::Ptr BundleAdjustment::get_optimizer(const OptimizerType opt_type,
                                               const TypeReprojectionError type_err,
                                               const Config& cfg) const {
    switch (opt_type) 
    {
        case OptimizerType::BA_CERES:
            return std::make_shared<CeresOptimizerReprojection>(type_err, cfg);
        case OptimizerType::BA_CERES_ICP:
            return std::make_shared<CeresOptimizerICP>(cfg);
        default:
            throw CuPhotoException("Error: Unknown type of optimization!");
    }
}


void BundleAdjustment::build_problem(const VisibilityGraph& vis_graph,
                                     const std::vector<Landmark::Ptr>& landmarks,
                                     const std::vector<KeyFrame::Ptr>& frames,
                                     const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                     const Camera::Ptr camera) {
    optimizer->build_blocks_reprojection(vis_graph, landmarks, frames, feat_pts, camera);
}


void BundleAdjustment::build_problem(const std::unordered_map<i32, ConnectionPoints>& pts3D,
                                     const std::vector<MatchAdjacent>& ma,
                                     const std::vector<KeyFrame::Ptr>& frames) {
    optimizer->build_blocks_icp(pts3D, ma, frames);
}


void BundleAdjustment::solve(std::vector<Landmark::Ptr>& landmarks, 
                             std::vector<KeyFrame::Ptr>& frames,
                             const Config& cfg) {
    const i32 num_iteration = cfg.get<i32>("motion.optimizer.num_iterations", 150);
    const i32 num_thread = cfg.get<i32>("motion.optimizer.num_thread", 2);
    const bool report = static_cast<bool>(cfg.get<i32>("motion.optimizer.full_report", 1));
    optimizer->optimize(num_iteration, num_thread, report);
    optimizer->store_result(landmarks, frames);
}


void BundleAdjustment::solve(std::vector<KeyFrame::Ptr>& frames,
                             const Config& cfg) {
    const i32 num_iteration = cfg.get<i32>("motion.optimizer.num_iterations", 150);
    const i32 num_thread = cfg.get<i32>("motion.optimizer.num_thread", 2);
    const bool report = static_cast<bool>(cfg.get<i32>("motion.optimizer.full_report", 1));
    optimizer->optimize(num_iteration, num_thread, report);
    optimizer->store_result(frames);
}

};
