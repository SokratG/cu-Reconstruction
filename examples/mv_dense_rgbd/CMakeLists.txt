cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# gflags
find_package(GFlags REQUIRED)
# glog
find_package(Glog REQUIRED)
# photogrammetry library
find_library(CUPHOTO_LIB cuphoto_lib)

include_directories("${PROJECT_SOURCE_DIR}/modules/include")
include_directories("${PROJECT_SOURCE_DIR}/examples/common")
include_directories(${GFLAGS_INCLUDE_DIRS})
include_directories(${GLOG_INCLUDE_DIRS})

set(SRC_DENSE_PC "mv_dense_pointcloud.cpp" "${PROJECT_SOURCE_DIR}/examples/common/rgbd_dataset.cpp")

add_executable(mv_dense_pc ${SRC_DENSE_PC})

target_link_libraries(mv_dense_pc cuphoto_lib)
target_link_libraries(mv_dense_pc ${GFLAGS_LIBRARIES})
target_link_libraries(mv_dense_pc ${GLOG_LIBRARIES})
target_link_libraries(mv_dense_pc stdc++fs)