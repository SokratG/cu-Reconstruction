cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# gflags
find_package(GFlags REQUIRED)
# glog
find_package(Glog REQUIRED)
# photogrammetry library
find_library(CUREC_LIB curec_lib)

include_directories("${PROJECT_SOURCE_DIR}/modules/include")
include_directories("${PROJECT_SOURCE_DIR}/examples/common")
include_directories(${GFLAGS_INCLUDE_DIRS})
include_directories(${GLOG_INCLUDE_DIRS})

set(SRC_DENSE_PC "dense_pointcloud.cpp" "${PROJECT_SOURCE_DIR}/examples/common/rgbd_dataset.cpp")

add_executable(dense_pc ${SRC_DENSE_PC})

target_link_libraries(dense_pc curec_lib)
target_link_libraries(dense_pc ${GFLAGS_LIBRARIES})
target_link_libraries(dense_pc ${GLOG_LIBRARIES})
target_link_libraries(dense_pc stdc++fs)