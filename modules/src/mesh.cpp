#include "mesh.hpp"
#include "point_cloud_types.hpp"

#include <pcl/io/ply_io.h>


namespace cuphoto {

std::vector<Mesh::NormalColorPoint> Mesh::vertices() const {
    return mesh_vertices;
}


void Mesh::vertices(const std::vector<Mesh::NormalColorPoint>& _mesh_vertices) {
    mesh_vertices = _mesh_vertices;
}

std::vector<Mesh::Triangle> Mesh::polygons() const {
    return mesh_polygons;
}

void Mesh::polygons(const std::vector<Mesh::Triangle>& _mesh_polygons) {
    mesh_polygons = _mesh_polygons;
}

void Mesh::store_to_ply(const std::string& ply_filepath) const {
    pcl::PolygonMeshPtr pcl_poly_mesh(new pcl::PolygonMesh);

    PointCloudCNPtr temp_pcl_pc(new PointCloudCN);
    temp_pcl_pc->points.resize(mesh_vertices.size());
    
    for (auto idx = 0; idx < mesh_vertices.size(); ++idx) {
        const auto point = mesh_vertices.at(idx);
        PointTCN pt;
        pt.x = point(0); pt.y = point(1); pt.z = point(2);
        pt.r = point(3); pt.g = point(4); pt.b = point(5);
        pt.normal_x = point(6); pt.normal_y = point(7); pt.normal_z = point(8);
        pt.curvature = point(9);
        temp_pcl_pc->points.at(idx) = pt;
    }

    pcl::toPCLPointCloud2(*temp_pcl_pc, pcl_poly_mesh->cloud);
    pcl_poly_mesh->polygons.resize(mesh_polygons.size());
    for (auto idx = 0; idx < mesh_polygons.size(); ++idx) {
        pcl::Vertices v;
        // add indices in triangle
        const Mesh::Triangle& tri = mesh_polygons.at(idx);
        v.vertices.push_back(tri(0));
        v.vertices.push_back(tri(1));
        v.vertices.push_back(tri(2));
        pcl_poly_mesh->polygons.at(idx) = v;
    }

    pcl::io::savePLYFileBinary(ply_filepath, *pcl_poly_mesh);
}


};