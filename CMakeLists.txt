cmake_minimum_required(VERSION 3.22)

project(cuReconstruction LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}" "${CMAKE_PREFIX_PATH}")

add_subdirectory(modules)
add_subdirectory(examples)
add_subdirectory(app)