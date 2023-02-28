#include "surfel_surface_reconstruction.hpp"
#include "point_cloud_utility.hpp"

#include "cp_exception.hpp"

#include <pcl/surface/surfel_smoothing.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/impl/mls.hpp>

#include <glog/logging.h>


namespace cuphoto {

using SurfelT = pcl::PointXYZRGBNormal;
using SurfelCloud = pcl::PointCloud<SurfelT>;
using SurfelCloudPtr = pcl::PointCloud<SurfelT>::Ptr;


struct SurfaceData {
    PointCloudCPtr input;
    SurfelCloudPtr sfcloud;
    pcl::PolygonMeshPtr poly_mesh;
};

SurfelSurface::SurfelSurface(const Config& cfg) {
    mls_radius = cfg.get<r32>("surface.surfels.mls.radius");
    polynomial_order = cfg.get<i32>("surface.surfels.mls.polynomial_order");

    triangle_search_radius = cfg.get<r32>("surface.surfels.gp3.search_radius");
    mu = cfg.get<r32>("surface.surfels.gp3.mu");
    max_nn = cfg.get<i32>("surface.surfels.gp3.max_nn");
    max_surf_angle = cfg.get<r32>("surface.surfels.gp3.max_surf_angle");
    min_angle = cfg.get<r32>("surface.surfels.gp3.min_angle");
    max_angle = cfg.get<r32>("surface.surfels.gp3.max_angle");
    normal_consistency = static_cast<bool>(cfg.get<i32>("surface.surfels.gp3.normal_consistency", 1));
}

void SurfelSurface::reconstruct_surface(const cudaPointCloud::Ptr cuda_pc) {
    if (cuda_pc->get_num_points() <= 0) {
        throw CuPhotoException("The point cloud data is empty! Can't run surfel surface reconstruction pipeline");
    }
    SurfaceData sd;
    sd.input = cuda_pc_to_pcl(cuda_pc);
    
    sd.sfcloud = SurfelCloudPtr(new SurfelCloud);
    build_surfel_point_cloud(sd);

    triangulate(sd);

    build_mesh(sd);
}


void SurfelSurface::build_mesh(SurfaceData& sd) {
    if (sd.poly_mesh->polygons.size() == 0) {
        throw CuPhotoException("The mesh polygons is empty! The data will not be saved");
    }
    const pcl::PolygonMeshPtr poly_mesh = sd.poly_mesh;
    PointCloudCNPtr temp_pcl_pc(new PointCloudCN);
    pcl::fromPCLPointCloud2(poly_mesh->cloud, *temp_pcl_pc);

    std::vector<Mesh::NormalColorPoint> points(temp_pcl_pc->points.size());
    for (auto idx = 0; idx < temp_pcl_pc->points.size(); ++idx) {
        PointTCN point = temp_pcl_pc->points.at(idx);
        Mesh::NormalColorPoint pt;
        pt << point.x, point.y, point.z, 
              static_cast<r32>(point.r),
              static_cast<r32>(point.g),
              static_cast<r32>(point.b),
              point.normal_x, point.normal_y, point.normal_z, point.curvature;
        points.at(idx) = pt;
    }

    std::vector<Mesh::Triangle> indices(poly_mesh->polygons.size());
    for (auto idx = 0; idx < poly_mesh->polygons.size(); ++idx) {
        Mesh::Triangle tri(poly_mesh->polygons.at(idx).vertices.at(0),
                           poly_mesh->polygons.at(idx).vertices.at(1),
                           poly_mesh->polygons.at(idx).vertices.at(2));
        indices.at(idx) = tri;
    }

    this->surface_mesh.vertices(points);
    this->surface_mesh.polygons(indices);
}


void SurfelSurface::build_surfel_point_cloud(SurfaceData& sd) {
    LOG(INFO) << "Start building surfels point cloud";
    
    pcl::MovingLeastSquares<PointTC, SurfelT> mls;

    pcl::search::KdTree<PointTC>::Ptr search_tree(new pcl::search::KdTree<PointTC>);

    mls.setSearchMethod(search_tree);
    mls.setSearchRadius(mls_radius);
    mls.setComputeNormals(true);
    mls.setSqrGaussParam(mls_radius * mls_radius);
    const bool fitting = polynomial_order > 1;
    mls.setPolynomialFit(fitting);
    mls.setPolynomialOrder(polynomial_order);

    mls.setInputCloud(sd.input);
    mls.process(*sd.sfcloud);
}


void SurfelSurface::triangulate(SurfaceData& sd) {
    LOG(INFO) << "Triangulate the mesh from point cloud";
    pcl::search::KdTree<SurfelT>::Ptr search_tree(new pcl::search::KdTree<SurfelT>);
    search_tree->setInputCloud(sd.sfcloud);

    pcl::GreedyProjectionTriangulation<SurfelT> gp3;
    pcl::PolygonMeshPtr triangles(new pcl::PolygonMesh);

    gp3.setSearchRadius(triangle_search_radius);


    gp3.setMu(mu);
    gp3.setMaximumNearestNeighbors(max_nn);
    gp3.setMaximumSurfaceAngle(max_surf_angle);
    gp3.setMinimumAngle(min_angle);
    gp3.setMaximumAngle(max_angle);
    gp3.setNormalConsistency(normal_consistency);

    gp3.setInputCloud(sd.sfcloud);
    gp3.setSearchMethod(search_tree);
    gp3.reconstruct(*triangles);

    sd.poly_mesh = triangles;
}


}