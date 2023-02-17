## Cuda Photogrammetry

### Structure from motion
1 | 2 | 3 |
:------:|:-------:|:------:|
![](gitresource/figures/B20.jpg)  |  ![](gitresource/figures/B21.jpg) | ![](gitresource/figures/B22.jpg) |

| Sparse point cloud |
:--------:|
|![](gitresource/figures/face_point_cloud.png) |
<b></b>

### Multiple view RGBD dense point cloud
1 | 2 | 3 |
:------:|:-------:|:------:|
![](gitresource/figures/00001-color.png)  |  ![](gitresource/figures/00002-color.png) | ![](gitresource/figures/00003-color.png) |

| Dense point cloud |
:--------:|
|![](gitresource/figure/table_point_cloud.png) |

### TODO ...

### TODO: add description


### Build project
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build build -j $(nproc)
```

### Tools:
* G++-8
* CMake>=3.22
* Cuda>=10.2
* TensorRT>=8.2

### Used third-party library:
* [Eigen 3.3.4](https://eigen.tuxfamily.org)
* [OpenCV 4.5.5](https://github.com/opencv/opencv)
* [Boost 1.65.1](https://www.boost.org/)
* [PCL 1.8](https://pointclouds.org)
* [Glog 0.5.0](https://github.com/google/glog)
* [Gflags 2.2.1](https://github.com/gflags/gflags)
* [Ceres-Solver](http://ceres-solver.org/)
* [Sophus](https://github.com/strasdat/Sophus)
* [Pangolin](https://github.com/stevenlovegrove/Pangolin)
* [CudaSift](https://github.com/Celebrandil/CudaSift)

