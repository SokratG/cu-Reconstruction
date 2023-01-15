#!/usr/bin/env bash

mkdir build
cd build
# TODO: add git init submodule with 3dparty(CudaSift) !
# !!!
cmake -DCMAKE_BUILD_TYPE=Release ..
cd ..
cmake --build build -j $(nproc)
