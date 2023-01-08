#include "landmark.hpp"
#include "utils.hpp"


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

void Landmark::set_observation(const std::unique_ptr<Feature> feature) {
    std::unique_lock<std::mutex> lck(data_mutex);
    observation = std::make_shared<Feature>(*feature);
}

std::shared_ptr<Feature> Landmark::landmark() const {
    std::unique_lock<std::mutex> lck(data_mutex);
    return observation;
}

Landmark::Ptr Landmark::create_landmark(const Vec3& position) {
    const uuid id = UUID::gen();
    Landmark::Ptr landmark = std::shared_ptr<Landmark>(new Landmark(id, position));
    return landmark;
}

Landmark::Ptr Landmark::create_landmark() {
    const uuid id = UUID::gen();
    Landmark::Ptr landmark = std::shared_ptr<Landmark>(new Landmark);
    landmark->id = id;
    return landmark;
}

};