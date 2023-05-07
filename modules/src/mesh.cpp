#include "mesh.hpp"
#include "point_cloud_types.hpp"
#include "utils.hpp"

#include <pcl/io/ply_io.h>

namespace cuphoto {


Mesh::Mesh(const std::vector<NormalColorPoint>& data_pts) {
    set_vertices_data(data_pts);
}


void Mesh::set_vertices_data(const std::vector<NormalColorPoint>& data_pts) {
    for (const auto& pt : data_pts) {
        Mesh::Point p(pt(0), pt(1), pt(2));
        Mesh::Color c(pt(3), pt(4), pt(5));
        Mesh::Normal n(pt(6), pt(7), pt(8), pt(9));
        mesh_vertices.emplace_back(p);
        mesh_colors.emplace_back(c);
        mesh_normals.emplace_back(n);
    }
}

std::vector<Mesh::Point> Mesh::vertices() const {
    return mesh_vertices;
}

std::vector<Mesh::Color> Mesh::colors() const {
    return mesh_colors;
}

std::vector<Mesh::Normal> Mesh::normals() const {
    return mesh_normals;
}


void Mesh::vertices(const std::vector<Mesh::Point>& _mesh_vertices) {
    mesh_vertices = _mesh_vertices;
}

void Mesh::colors(const std::vector<Mesh::Color>& _mesh_colors) {
    mesh_colors = _mesh_colors;
}

void Mesh::normals(const std::vector<Mesh::Normal>& _mesh_normals) {
    mesh_normals = _mesh_normals;
}

std::vector<Mesh::Triangle> Mesh::polygons() const {
    return mesh_polygons;
}

void Mesh::polygons(const std::vector<Mesh::Triangle>& _mesh_polygons) {
    mesh_polygons = _mesh_polygons;
}

void Mesh::store_to_ply(const std::string& ply_filepath) const {
    if (mesh_normals.empty())
        write_mesh_to_ply(ply_filepath, mesh_vertices, mesh_polygons);
    else
        save_from_pcl(ply_filepath);
}

void Mesh::save_from_pcl(const std::string& ply_filepath) const {
    pcl::PolygonMeshPtr pcl_poly_mesh(new pcl::PolygonMesh);

    PointCloudCNPtr temp_pcl_pc(new PointCloudCN);
    temp_pcl_pc->points.resize(mesh_vertices.size());
    
    for (auto idx = 0; idx < mesh_vertices.size(); ++idx) {
        PointTCN pt;
        const auto point = mesh_vertices.at(idx);
        pt.x = point(0); pt.y = point(1); pt.z = point(2);
        const auto color = mesh_colors.at(idx);
        pt.r = color(0); pt.g = color(1); pt.b = color(2);
        const auto normals = mesh_normals.at(idx);
        pt.normal_x = normals(0); pt.normal_y = normals(1); pt.normal_z = normals(2);
        pt.curvature = normals(3);
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