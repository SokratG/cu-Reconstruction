cmake_minimum_required(VERSION 3.22)

# https://github.com/dusty-nv/jetson-utils
project(cudaUtils)
set(cudaUtils_VERSION_MAJOR 1)
set(cudaUtils_VERSION_MINOR 0)
set(cudaUtils_VERSION_PATCH 0)

set(CPACK_PACKAGE_VERSION_MAJOR "${cudaUtils_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${cudaUtils_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${cudaUtils_VERSION_PATCH}")
set(CPACK_GENERATOR "ZIP")
include(CPack)

find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_NVCC_FLAGS
    ${CMAKE_NVCC_FLAGS};
    -lineinfo
    -ccbin
    # -O3
    -D_FORCE_INLINES
)


set(cuda_sources
  cudaMath.cuh
  cudaUtility.cuh
  cudaVector.cuh
  cudaMappedMemory.cuh
  cudaResize.cuh
  cudaResize.cu
  cudaGrayscale.cuh
  cudaGrayscale.cu
  cudaFilterMode.cuh
  cudaFilterMode.cu
  cudaColormap.cuh
  cudaColormap.cu
  tensorConvert.cuh
  tensorConvert.cu
)

set(sources
    Mutex.inl
    Event.inl
    Process.cpp
    randInt8Calibrator.cpp
    filesystem.cpp
    timespec.cpp
    logging.cpp
    tensorNet.cpp
)


include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR} 
)

#SET(CUDA_SEPARABLE_COMPILATION ON)

add_library(cudaUtils SHARED ${cuda_sources} ${sources})

set_target_properties(cudaUtils PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"
  OUTPUT_NAME "cudaUtils"
  CUDA_STANDARD 14
  CUDA_SEPARABLE_COMPILATION ON
)

install(TARGETS cudaUtils
        LIBRARY DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
        PUBLIC_HEADER DESTINATION ${CMAKE_INCLUDE_OUTPUT_DIRECTORY})

# set linker options
target_link_libraries(cudaUtils ${CUDA_cudadevrt_LIBRARY})
target_link_libraries(cudaUtils nvinfer nvinfer_plugin nvcaffe_parser)
