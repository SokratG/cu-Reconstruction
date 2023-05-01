#include <stdlib.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "config.hpp"
#include "cp_exception.hpp"
#include "point_cloud_dataset.hpp"
#include "tsdf_surface_reconstruction.hpp"

DEFINE_string(config_path, "./data/tsdf_surface_parameters.yaml", "file path with parameters");

int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    
    cuphoto::Config cfg;
    if (cfg.set_parameter_file(FLAGS_config_path) == false)
        return EXIT_FAILURE;

    cuphoto::PointCloudDataset pc_dataset(cfg);

    const auto cu_pc = pc_dataset.get_next();

    const std::string store_path = cfg.get<std::string>("mesh_store.path");


    try {
        cuphoto::TSDFSurface tsdf(cfg);

        tsdf.reconstruct_surface(cu_pc);

        // const auto mesh = tsdf.get_mesh();

        // mesh.store_to_ply(store_path);

    } catch(cuphoto::CuPhotoException& photo_ex) {
        LOG(ERROR) << photo_ex.what();
        return EXIT_FAILURE;
    } catch (std::exception& ex) {
        LOG(ERROR) << ex.what();
        return EXIT_FAILURE;
    } catch(...) {
        return EXIT_FAILURE;
    }
    

    return EXIT_SUCCESS;
}