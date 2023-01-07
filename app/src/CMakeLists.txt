cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# gflags
find_package(GFlags REQUIRED)
# glog
find_package(Glog REQUIRED)
# reconstruction library
find_library(CUREC_LIB curec_lib)

include_directories("${PROJECT_SOURCE_DIR}/modules/include")
include_directories(${GFLAGS_INCLUDE_DIRS})
include_directories(${GLOG_INCLUDE_DIRS})

file(GLOB SRC_MAIN_APP "*.c*")

add_executable(curec_app ${SRC_MAIN_APP})

target_link_libraries(curec_app curec_lib)
target_link_libraries(curec_app ${GFLAGS_LIBRARIES})
target_link_libraries(curec_app ${GLOG_LIBRARIES})