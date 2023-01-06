#include "landmark.hpp"

namespace curec {

Landmark::Landmark(const uuid& _id, const Vec3& _position) : id(_id), position(_position) {

}

bool Landmark::outlier() const {
    return is_outlier;
}

void Landmark::outlier(const bool v) {
    is_outlier = v;
}

Vec3 Landmark::pose() const {
    std::unique_lock<std::mutex> lck(data_mutex);
    return position;
}

void Landmark::pose(const Vec3& _position) {
    std::unique_lock<std::mutex> lck(data_mutex);
    position = _position;
};

void Landmark::add_observation(const std::shared_ptr<Feature> feature) {
    std::unique_lock<std::mutex> lck(data_mutex);
    observations.push_back(feature);
}

void Landmark::remove_observation(const std::shared_ptr<Feature> feature) {
    std::unique_lock<std::mutex> lck(data_mutex);
    for (auto iter = observations.begin(); iter != observations.end(); iter++) {
        if (iter->lock() == feature) {
            observations.erase(iter);
            break;
        }
    }
}

std::list<std::weak_ptr<Feature>> Landmark::observation() const {
    std::unique_lock<std::mutex> lck(data_mutex);
    return observations;
}

};