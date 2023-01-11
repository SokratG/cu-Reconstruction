#include "landmark.hpp"
#include "utils.hpp"


namespace curec {

Landmark::Landmark(const uuid& _id, const Vec3& _position, const Vec3f& _color) : 
                  id(_id), position(_position), landmark_color(_color) {

}

Vec3 Landmark::pose() const {
    std::unique_lock<std::mutex> lck(data_mutex);
    return position;
}

void Landmark::pose(const Vec3& _position) {
    std::unique_lock<std::mutex> lck(data_mutex);
    position = _position;
};

Vec3f Landmark::color() const {
    std::unique_lock<std::mutex> lck(data_mutex);
    return landmark_color;
}

void Landmark::color(const Vec3f& _color) {
    std::unique_lock<std::mutex> lck(data_mutex);
    landmark_color = _color;
}

void Landmark::set_observation(const std::shared_ptr<Feature> feature) {
    std::unique_lock<std::mutex> lck(data_mutex);
    observation = feature;
}

std::shared_ptr<Feature> Landmark::landmark() const {
    std::unique_lock<std::mutex> lck(data_mutex);
    return observation;
}

Landmark::Ptr Landmark::create_landmark(const Vec3& position, const Vec3f& color) {
    const uuid id = UUID::gen();
    Landmark::Ptr landmark = std::shared_ptr<Landmark>(new Landmark(id, position, color));
    return landmark;
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