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

set(SRC_SPARSE "sparse_reconstruction.cpp" 
               "${PROJECT_SOURCE_DIR}/examples/common/rgbd_dataset.cpp"
               "${PROJECT_SOURCE_DIR}/examples/common/rgb_dataset.cpp")

add_executable(sparse ${SRC_SPARSE})

target_link_libraries(sparse cuphoto_lib)
target_link_libraries(sparse ${GFLAGS_LIBRARIES})
target_link_libraries(sparse ${GLOG_LIBRARIES})
