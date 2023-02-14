cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Boost_NO_WARN_NEW_VERSIONS ON)


if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()


# CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)
set(CMAKE_NVCC_FLAGS
    ${CMAKE_NVCC_FLAGS};
    -Xcompiler
    -fopenmp
    # -O3
    -use_fast_math)

set(CMAKE_CUDA_FLAGS)


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
find_package(PCL 1.10 REQUIRED COMPONENTS common filters)
# CudaSift
# find_library(CUDA_SIFT 
#             NAMES cudaSift
#             PATHS ${CMAKE_BINARY_DIR}
#             PATH_SUFFIXES 3dparty/CudaSift
#             NO_DEFAULT_PATH)

# includes
include_directories("${PROJECT_SOURCE_DIR}/modules/include")
include_directories("${PROJECT_SOURCE_DIR}/modules/include/cuda")
include_directories("${PROJECT_SOURCE_DIR}/3dparty")
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${GLOG_INCLUDE_DIRS})

add_definitions(${PCL_DEFINITIONS})
add_definitions(-D_USE_MATH_DEFINES)

file(GLOB SRC_CUPHOTO_LIB "src/*.cpp" "src/cuda/*.cu")

file(GLOB HEADER_CUPHOTO "${PROJECT_SOURCE_DIR}/modules/include/*.h*" 
                         "${PROJECT_SOURCE_DIR}/modules/include/cuda/*.cuh")

add_library(cuphoto_lib SHARED ${SRC_CUPHOTO_LIB})

set_target_properties(cuphoto_lib PROPERTIES
                PUBLIC_HEADER "${HEADER_CUPHOTO}"
                OUTPUT_NAME "cuphoto_lib" 
		        CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS cuphoto_lib
        LIBRARY DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        PUBLIC_HEADER DESTINATION ${CMAKE_INCLUDE_OUTPUT_DIRECTORY})

set_property(TARGET cuphoto_lib PROPERTY CUDA_STANDARD 14)


target_link_libraries(cuphoto_lib rt)
target_link_libraries(cuphoto_lib cudaSift)
target_link_libraries(cuphoto_lib cudaUtils)
target_link_libraries(cuphoto_lib Sophus::Sophus)
target_link_libraries(cuphoto_lib ${CERES_LIBRARIES})
target_link_libraries(cuphoto_lib ${CUDA_LIBRARIES})
target_link_libraries(cuphoto_lib ${PCL_LIBRARIES})
target_link_libraries(cuphoto_lib ${OpenCV_LIBRARIES})
target_link_libraries(cuphoto_lib ${GLOG_LIBRARIES})
target_link_libraries(cuphoto_lib stdc++fs)


target_compile_options(cuphoto_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_NVCC_FLAGS}>)