#ifndef CUREC_LIB_BUNDLE_ADJUSTMENT_HPP
#define CUREC_LIB_BUNDLE_ADJUSTMENT_HPP

#include "optimizer.hpp"
#include <memory>
#include <vector>

namespace curec {

enum class OptimizerType {
    BA_CERES = 0,
    BA_UNKNOWN
};

struct BAParam {
    i32 num_thread = 2;
    i32 num_iteration = 100;
    bool report = true;
};


class BundleAdjustment {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<BundleAdjustment>;

    BundleAdjustment(const OptimizerType opt_type, const TypeReprojectionError type_err);
    void build_problem(const VisibilityGraph& landmarks, const std::vector<KeyFrame::Ptr>& frames, const Camera::Ptr camera);
    void solve(VisibilityGraph& landmarks, std::vector<KeyFrame::Ptr>& frames, const BAParam& ba_param = BAParam());
protected:
    Optimizer::Ptr get_optimizer(const OptimizerType opt_type, const TypeReprojectionError type_err) const;
private:
    Optimizer::Ptr optimizer;
};


};


#endif