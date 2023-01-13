#include "bundle_adjustment.hpp"
#include "ceres_optim.hpp"
#include "cr_exception.hpp"

namespace curec {

BundleAdjustment::BundleAdjustment(const OptimizerType opt_type,
                                   const TypeReprojectionError type_err) {
    optimizer = get_optimizer(opt_type, type_err);
}

Optimizer::Ptr BundleAdjustment::get_optimizer(const OptimizerType opt_type,
                                               const TypeReprojectionError type_err) const {
    switch (opt_type) 
    {
        case OptimizerType::BA_CERES:
            return std::make_shared<CeresOptimizer>(type_err);
        default:
            throw CuRecException("Error: Unknown type of optimization!");
    }
}


void BundleAdjustment::build_problem(const VisibilityGraph& landmarks,
                                     const std::vector<KeyFrame::Ptr>& frames,
                                     const Camera::Ptr camera) {
    optimizer->build_blocks(landmarks, frames, camera);
}


void BundleAdjustment::solve(VisibilityGraph& landmarks, std::vector<KeyFrame::Ptr>& frames, const BAParam& ba_param) {
    optimizer->optimize(ba_param.num_iteration, ba_param.num_thread, ba_param.report);
    optimizer->store_result(landmarks, frames);
}

};
