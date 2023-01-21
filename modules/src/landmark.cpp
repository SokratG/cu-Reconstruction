#include "landmark.hpp"
#include "keyframe.hpp"
#include "utils.hpp"


namespace cuphoto {

Landmark::Landmark(const uuid& _id, const Vec3& _position, const Vec3f& _color) : 
                  id(_id), position(_position), landmark_color(_color) {

}

Vec3 Landmark::pose() const {
    return position;
}

void Landmark::pose(const Vec3& _position) {
    position = _position;
};

Vec3f Landmark::color() const {
    return landmark_color;
}

void Landmark::color(const Vec3f& _color) {
    landmark_color = _color;
}

void Landmark::observation(const std::shared_ptr<Feature> feature) {
    observation_pt = feature;
}

std::shared_ptr<Feature> Landmark::observation() const {
    return observation_pt;
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


Landmark::Ptr make_landmark(const Feature::Ptr feat_pt, const Vec3& position_world) {
    const auto pt2 = feat_pt->position.pt;
    const Vec3f color = feat_pt->frame.lock()->get_color(pt2);
    Landmark::Ptr landmark = Landmark::create_landmark(position_world, color);
    landmark->observation(feat_pt);
    return landmark;
}

};