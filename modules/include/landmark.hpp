#ifndef CUREC_LIB_LANDMARK_HPP
#define CUREC_LIB_LANDMARK_HPP

#include "types.hpp"
#include "feature.hpp"
#include <opencv2/core/core.hpp>
#include <memory>
#include <mutex>
#include <unordered_map>


namespace curec {

class Landmark {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<Landmark>;


    // factory build mode, assign id
    static std::shared_ptr<Landmark> create_landmark(const Vec3& position, const Vec3f& color);
    static std::shared_ptr<Landmark> create_landmark(const Vec3& position);
    static std::shared_ptr<Landmark> create_landmark();

    
    Vec3 pose() const;
    void pose(const Vec3& position);
    Vec3f color() const;
    void color(const Vec3f& color);
    void observation(const std::shared_ptr<Feature> feature);
    std::shared_ptr<Feature> observation() const;

public:
    uuid id;

protected:
    Landmark() {}
    Landmark(const uuid& id, const Vec3& position, const Vec3f& color = Vec3f(0, 0, 0));
private:
    mutable std::mutex data_mutex;
    Vec3 position = Vec3::Zero();
    Vec3f landmark_color = Vec3f(255, 255, 255);
    std::shared_ptr<Feature> observation_pt;
};


// 
using VisibilityGraph = std::unordered_map<ui32, std::unordered_map<ui64, Landmark::Ptr>>;

};

#endif // CUREC_LANDMARK_HPP