#ifndef CUREC_LIB_LANDMARK_HPP
#define CUREC_LIB_LANDMARK_HPP

#include "types.hpp"
#include "feature.hpp"
#include <opencv2/core/core.hpp>
#include <memory>
#include <mutex>
#include <list>


namespace curec {

class Landmark {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    using Ptr = std::shared_ptr<Landmark>;

    Landmark() {}
    Landmark(const uuid& id, const Vec3& position);

    bool outlier() const;
    void outlier(const bool v);
    Vec3 pose() const;
    void pose(const Vec3& position);
    void add_observation(const std::shared_ptr<Feature> feature);
    void remove_observation(const std::shared_ptr<Feature> feature);
    std::list<std::weak_ptr<Feature>> observation() const;

public:
    uuid id;

private:
    mutable std::mutex data_mutex;
    Vec3 position;
    std::list<std::weak_ptr<Feature>> observations;
    bool is_outlier;
};

};

#endif // CUREC_LANDMARK_HPP