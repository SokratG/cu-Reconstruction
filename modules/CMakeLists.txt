cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Boost_NO_WARN_NEW_VERSIONS ON)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

#opencv
find_package(OpenCV REQUIRED)
# eigen
find_package(Eigen3 REQUIRED)
# glog
find_package(Glog REQUIRED)
# sophus
find_package(Sophus REQUIRED)
# ceres
find_package(Ceres REQUIRED)
# boost
find_package(Boost REQUIRED)
# PCL
find_package(PCL 1.10 REQUIRED)
# cuda
find_package(CUDA REQUIRED)

# includes
include_directories("${PROJECT_SOURCE_DIR}/modules/include")
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${GLOG_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})


add_definitions(-D_USE_MATH_DEFINES)

file(GLOB SRC_CUREC_LIB "src/*.c*")
file(GLOB HEADER_CUREC "${PROJECT_SOURCE_DIR}/modules/include/*.h*")

add_library(curec_lib SHARED ${SRC_CUREC_LIB})

install(TARGETS curec_lib
        LIBRARY DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        PUBLIC_HEADER DESTINATION ${CMAKE_INCLUDE_OUTPUT_DIRECTORY})

set_target_properties(curec_lib PROPERTIES PUBLIC_HEADER "${HEADER_CUREC}")

target_link_libraries(curec_lib rt)
target_link_libraries(curec_lib Sophus::Sophus)
target_link_libraries(curec_lib ${CERES_LIBRARIES})
target_link_libraries(curec_lib ${OpenCV_LIBRARIES})
target_link_libraries(curec_lib ${GLOG_LIBRARIES})
target_link_libraries(curec_lib ${PCL_LIBRARIES})
