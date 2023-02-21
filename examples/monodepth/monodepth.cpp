#include <stdlib.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "config.hpp"
#include "rgb_dataset.hpp"

DEFINE_string(config_path, "./data/monodepth_parameters.yaml", "file path with parameters");


int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    cuphoto::Config cfg;
    if (cfg.set_parameter_file(FLAGS_config_path) == false)
        return EXIT_FAILURE;

    cuphoto::RGBDataset rgb_dataset(cfg.get<std::string>("dataset.path"));

    return EXIT_SUCCESS;

}