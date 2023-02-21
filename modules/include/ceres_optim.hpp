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
    CeresCameraModel(const SE3& camera_pose);

    SE3 pose() const;
    r64 raw_camera_param[7];
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


class ICPErrorPose
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<ICPErrorPose>;

    ICPErrorPose(const Vec3& _src_pt, const Vec3& _dst_pt) : 
                 src_pt(_src_pt), dst_pt(_dst_pt) {

    }

    // camera : 7 dims array
    // [0-3] : quaternion rotation
    // [4-6] : translation
    // predictions : 3D prediction as a diff between src and dst point
    template<typename T>
    bool operator()(const T* camera,
                    T* residual) const {
        T P[3];
        const T world_point[3] {static_cast<T>(src_pt.x()), static_cast<T>(src_pt.y()), static_cast<T>(src_pt.z())};

        ceres::QuaternionRotatePoint(camera, world_point, P);  // UnitQuaternionRotatePoint?
        P[0] += camera[4];
        P[1] += camera[5];
        P[2] += camera[6];
        
        residual[0] = static_cast<T>(dst_pt.x()) - P[0];
        residual[1] = static_cast<T>(dst_pt.y()) - P[1];
        residual[2] = static_cast<T>(dst_pt.z()) - P[2];
        return true;
    }

    static ceres::CostFunction* create(const Vec3& src_pt, const Vec3& dst_pt) {
        return (new ceres::AutoDiffCostFunction<ICPErrorPose, 3, 7>(
            new ICPErrorPose(src_pt, dst_pt)
        ));
    }

private:
    Vec3 src_pt;
    Vec3 dst_pt;
};


class CeresOptimizerReprojection : public Optimizer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<CeresOptimizerReprojection>;

    CeresOptimizerReprojection(const TypeReprojectionError tre, const Config& cfg);
    void optimize(const i32 n_iteration, const i32 num_threads, const bool fullreport) override;

    void build_blocks_reprojection(const VisibilityGraph& vis_graph,
                                   const std::vector<Landmark::Ptr>& landmarks, 
                                   const std::vector<KeyFrame::Ptr>& frames,
                                   const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                   const Camera::Ptr camera) override;
    
    void store_result(std::vector<Landmark::Ptr>& landmarks, 
                      std::vector<KeyFrame::Ptr>& frames) override;
    void reset();

    void build_blocks_icp(const std::unordered_map<i32, ConnectionPoints>& pts3D,
                          const std::vector<MatchAdjacent>& ma, 
                          const std::vector<KeyFrame::Ptr>& frames) override;

    void store_result(std::vector<KeyFrame::Ptr>& frames) override;
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


class CeresOptimizerICP : public Optimizer {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<CeresOptimizerICP>;

    CeresOptimizerICP(const Config& cfg);
    void optimize(const i32 n_iteration, const i32 num_threads, const bool fullreport) override;
    void build_blocks_icp(const std::unordered_map<i32, ConnectionPoints>& pts3D,
                          const std::vector<MatchAdjacent>& ma, 
                          const std::vector<KeyFrame::Ptr>& frames) override;
    void store_result(std::vector<KeyFrame::Ptr>& frames) override;


    void build_blocks_reprojection(const VisibilityGraph& vis_graph,
                                   const std::vector<Landmark::Ptr>& landmarks, 
                                   const std::vector<KeyFrame::Ptr>& frames,
                                   const std::vector<std::vector<Feature::Ptr>>& feat_pts,
                                   const Camera::Ptr camera) override;
    
    void store_result(std::vector<Landmark::Ptr>& landmarks, 
                      std::vector<KeyFrame::Ptr>& frames) override;

    void reset();
private:
    void add_block(CeresCameraModel& ceres_camera, const Vec3& src_pt, const Vec3& dst_pt); 

private:
    std::shared_ptr<ceres::Problem> optim_problem;
    std::unordered_map<ui64, CeresCameraModel> ceres_cameras;
    r64 loss_width;
};


};


#endif