#ifndef CUPHOTO_LIB_CERES_OPTIM_HPP
#define CUPHOTO_LIB_CERES_OPTIM_HPP

#include "optimizer.hpp"
#include "utils.hpp"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <unordered_map>


namespace cuphoto {

struct CeresCameraModel {
    CeresCameraModel() = delete;
    CeresCameraModel(const SE3& camera_pose, const Mat3& K);

    Mat3 K() const;
    SE3 pose() const;
    r64 raw_camera_param[8];
    Vec2 camera_center;
};

struct CeresObservation {
    CeresObservation() = delete;
    CeresObservation(const Vec3& pt3d);

    Vec3 position() const;
    r64 obs[3];
};

class ReprojectionErrorPosePoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<ReprojectionErrorPosePoint>;


    ReprojectionErrorPosePoint(const Mat3& camera_matrix, const Vec2& observation_point) : 
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

        const r64 fx = K(0, 0);
        const r64 fy = K(1, 1);
        const r64 cx = K(0, 2);
        const r64 cy = K(1, 2);

        // project to camera image: p = K * P'
        T prediction[2];
        prediction[0] = fx * P[0] / P[2] + cx;
        prediction[1] = fy * P[1] / P[2] + cy;

        residual[0] = static_cast<T>(observation.x()) - prediction[0];
        residual[1] = static_cast<T>(observation.y()) - prediction[1];
        return true;
    }

    static ceres::CostFunction* create(const Mat3& camera_matrix, const Vec2& observation_point) {
        return (new ceres::AutoDiffCostFunction<ReprojectionErrorPosePoint, 2, 7, 3>(
            new ReprojectionErrorPosePoint(camera_matrix, observation_point)
        ));
    }

private:
    Vec2 observation;
    Mat3 K;
};



class ReprojectionErrorPose
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<ReprojectionErrorPose>;


    ReprojectionErrorPose(const Mat3& camera_matrix, const Vec3& world_point, const Vec2& observation_point) : 
                          world_pt(world_point), observation(observation_point), K(camera_matrix) {

    }

    // camera : 7 dims array
    // [0-3] : quaternion rotation
    // [4-6] : translation
    // predictions : 2D predictions with center of the image plane
    template<typename T>
    bool operator()(const T* camera,
                    T* residual) const {
        T P[3];
        const T world_point[3] {static_cast<T>(world_pt.x()), static_cast<T>(world_pt.y()), static_cast<T>(world_pt.z())};

        ceres::QuaternionRotatePoint(camera, world_point, P);  // UnitQuaternionRotatePoint?
        P[0] += camera[4];
        P[1] += camera[5];
        P[2] += camera[6];

        const r64 fx = K(0, 0);
        const r64 fy = K(1, 1);
        const r64 cx = K(0, 2);
        const r64 cy = K(1, 2);

        // project to camera image: p = K * P'
        T prediction[2];
        prediction[0] = fx * P[0] / P[2] + cx;
        prediction[1] = fy * P[1] / P[2] + cy;

        residual[0] = static_cast<T>(observation.x()) - prediction[0];
        residual[1] = static_cast<T>(observation.y()) - prediction[1];
        return true;
    }

    static ceres::CostFunction* create(const Mat3& camera_matrix, const Vec3& world_point, const Vec2& observation_point) {
        return (new ceres::AutoDiffCostFunction<ReprojectionErrorPose, 2, 7>(
            new ReprojectionErrorPose(camera_matrix, world_point, observation_point)
        ));
    }

private:
    Vec3 world_pt;
    Vec2 observation;
    Mat3 K;
};


class ReprojectionErrorFocalPose
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<ReprojectionErrorFocalPose>;


    ReprojectionErrorFocalPose(const Vec2& camera_center, const Vec2& observation_point) : 
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
        prediction[0] = f * P[0] / P[2] + center.x();
        prediction[1] = f * P[1] / P[2] + center.y();

        residual[0] = T(observation.x()) - prediction[0];
        residual[1] = T(observation.y()) - prediction[1];
        return true;
    }

    static ceres::CostFunction* create(const Vec2& camera_center, const Vec2& observation_point) {
        return (new ceres::AutoDiffCostFunction<ReprojectionErrorFocalPose, 2, 8, 3>(
            new ReprojectionErrorFocalPose(camera_center, observation_point)
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

    CeresOptimizer(const TypeReprojectionError tre, const r64 loss_width = 6.5);
    virtual void build_blocks(const VisibilityGraph& vis_graph,
                              const std::vector<Landmark::Ptr>& landmarks, 
                              const std::vector<KeyFrame::Ptr>& frames,
                              const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                              const Camera::Ptr camera) override;
    virtual void optimize(const i32 n_iteration, const i32 num_threads, const bool fullreport) override;
    virtual void store_result(std::vector<Landmark::Ptr>& landmarks, 
                              std::vector<KeyFrame::Ptr>& frames) override;
    void reset();
protected:
    void add_block(CeresCameraModel& ceres_camera, CeresObservation& landmark, 
                   const Vec2& observ_pt, const Mat3& K);
    ceres::CostFunction* get_cost_function(const Mat3& K, const Vec3& world_pt, const Vec2& pt);

private:
    std::shared_ptr<ceres::Problem> optim_problem;
    std::unordered_map<ui64, CeresCameraModel> ceres_cameras;
    std::unordered_map<ui64, CeresObservation> ceres_obseravations;
    r64 loss_width;
    TypeReprojectionError type_err;
};

};


#endif