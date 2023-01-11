#ifndef CUREC_LIB_LANDMARK_HPP
#define CUREC_LIB_LANDMARK_HPP

#include "types.hpp"
#include "feature.hpp"
#include <opencv2/core/core.hpp>
#include <memory>
#include <mutex>


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
    void set_observation(const std::shared_ptr<Feature> feature);
    std::shared_ptr<Feature> landmark() const;

public:
    uuid id;

protected:
    Landmark() {}
    Landmark(const uuid& id, const Vec3& position, const Vec3f& color = Vec3f(0, 0, 0));
private:
    mutable std::mutex data_mutex;
    Vec3 position;
    Vec3f landmark_color;
    std::shared_ptr<Feature> observation;
};

};

#endif // CUREC_LANDMARK_HPP