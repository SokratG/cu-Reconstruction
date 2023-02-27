#include "single_view_scene.hpp"


namespace cuphoto {


SingleViewScene::SingleViewScene(const Camera::Ptr _camera) : camera(_camera) {

}

cudaPointCloud::Ptr SingleViewScene::get_point_cloud() const {
    return cuda_pc;
}

bool SingleViewScene::store_to_ply(const std::string& ply_filepath) const {
    return cuda_pc->save_ply(ply_filepath);
}

};