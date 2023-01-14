#ifndef CUREC_LIB_CERES_OPTIM_HPP
#define CUREC_LIB_CERES_OPTIM_HPP

#include "optimizer.hpp"
#include "utils.hpp"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <unordered_map>

namespace curec {

struct CeresCameraModel {
    CeresCameraModel() = delete;
    CeresCameraModel(const SE3& camera_pose, const Mat3& K);

    Mat3 K() const;
    SE3 pose() const;
    r64 raw_camera_param[9];
    Vec2 camera_center;
};

struct CeresObservation {
    CeresObservation() = delete;
    CeresObservation(const Vec3& pt3d);

    Vec3 position() const;
    r64 obs[3];
};

class ReprojectionErrorRt
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<ReprojectionErrorRt>;


    ReprojectionErrorRt(const Mat3& camera_matrix, const Vec2& observation_point) : 
                        observation(observation_point), K(camera_matrix) {

    }

    // camera : 7 dims array
    // [0-3] : quaternion rotation
    // [4-6] : translation
    // point : 3D location
    // predictions : 2D predictions with center of the image plane
    template<typename T>
    bool operator()(const T* camera,
                    const T* point,
                    T* residual) const {
        T P[3];
        ceres::QuaternionRotatePoint(camera, point, P);  // UnitQuaternionRotatePoint?
        P[0] += camera[4];
        P[1] += camera[5];
        P[2] += camera[6];

        const T fx = static_cast<T>(K(0, 0));
        const T fy = static_cast<T>(K(1, 1));
        const T cx = static_cast<T>(K(0, 2));
        const T cy = static_cast<T>(K(1, 2));

        // project to camera image: p = K * P'
        T prediction[2];
        prediction[0] = fx * P[0] / P[2] + cx;
        prediction[1] = fy * P[0] / P[2] + cy;

        residual[0] = prediction[0] - T(observation.x());
        residual[1] = prediction[1] - T(observation.y());
        return true;
    }

    static ceres::CostFunction* create(const Mat3& camera_matrix, const Vec2& observation_point) {
        return (new ceres::AutoDiffCostFunction<ReprojectionErrorRt, 2, 7, 3>(
            new ReprojectionErrorRt(camera_matrix, observation_point)
        ));
    }

private:
    Vec2 observation;
    Mat3 K;
};


class ReprojectionErrorFocalRt
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<ReprojectionErrorFocalRt>;


    ReprojectionErrorFocalRt(const Vec2& camera_center, const Vec2& observation_point) : 
                             center(camera_center), observation(observation_point) {

    }

    // camera : 8 dims array
    // [0-3] : quaternion rotation
    // [4-6] : translation
    // [7] : focal length
    // point : 3D location
    // predictions : 2D predictions with center of the image plane
    template<typename T>
    bool operator()(const T* camera,
                    const T* point,
                    T* residual) const {
        T P[3];
        ceres::QuaternionRotatePoint(camera, point, P);  // UnitQuaternionRotatePoint?
        P[0] += camera[4];
        P[1] += camera[5];
        P[2] += camera[6];

        const T f = camera[7];

        // project to camera image: p = K * P'
        T prediction[2];
        prediction[0] = f * P[0] / (P[2] + 1e-7) + center.x();
        prediction[1] = f * P[0] / (P[2] + 1e-7) + center.y();

        residual[0] = prediction[0] - T(observation.x());
        residual[1] = prediction[1] - T(observation.y());
        return true;
    }

    static ceres::CostFunction* create(const Vec2& camera_center, const Vec2& observation_point) {
        return (new ceres::AutoDiffCostFunction<ReprojectionErrorFocalRt, 2, 8, 3>(
            new ReprojectionErrorFocalRt(camera_center, observation_point)
        ));
    }

private:
    Vec2 observation;
    Vec2 center;
};


class CeresOptimizer : public Optimizer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<CeresOptimizer>;

    CeresOptimizer(const TypeReprojectionError tre, const r64 loss_width = 1.0);
    virtual void build_blocks(const VisibilityGraph& landmarks,
                              const std::vector<KeyFrame::Ptr>& frames,
                              const Camera::Ptr camera) override;
    virtual void optimize(const i32 n_iteration, const i32 num_threads, const bool fullreport) override;
    virtual void store_result(VisibilityGraph& landmarks, std::vector<KeyFrame::Ptr>& frames) override;
    void reset();
protected:
    void add_block(CeresCameraModel& ceres_camera, CeresObservation& landmark, 
                   const Vec2& observ_pt, const Mat3& K);
    ceres::CostFunction* get_cost_function(const Mat3& K, const Vec2& pt);

private:
    std::shared_ptr<ceres::Problem> optim_problem;
    std::unordered_map<uuid, CeresCameraModel> ceres_cameras;
    std::unordered_map<uuid, CeresObservation> ceres_obseravations;
    r64 loss_width;
    TypeReprojectionError type_err;
};



};


#endif