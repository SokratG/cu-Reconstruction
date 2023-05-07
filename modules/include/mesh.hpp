#ifndef CUPHOTO_LIB_MESH_HPP
#define CUPHOTO_LIB_MESH_HPP

#include "types.hpp"

#include <vector>
#include <string>

namespace cuphoto {


class Mesh {
public:
    using Point = Vec3f;
    using Color = Vec3f;
    using Normal = Vec4f;
    using ColorPoint = Vec6f;
    using NormalColorPoint = Vec10f;
    using Triangle = Vec3i;
public:
    Mesh() = default;
    Mesh(const std::vector<NormalColorPoint>& data_pts);

    void set_vertices_data(const std::vector<NormalColorPoint>& data_pts);

    std::vector<Mesh::Point> vertices() const;
    std::vector<Mesh::Color> colors() const;
    std::vector<Mesh::Normal> normals() const;

    void colors(const std::vector<Mesh::Color>& mesh_colors);
    void vertices(const std::vector<Mesh::Point>& mesh_vertices);
    void normals(const std::vector<Mesh::Normal>& mesh_normals);

    std::vector<Mesh::Triangle> polygons() const;

    void polygons(const std::vector<Mesh::Triangle>& mesh_faces);

    void store_to_ply(const std::string& ply_filepath) const;
private:
    void save_from_pcl(const std::string& ply_filepath) const;
private:
    std::vector<Point> mesh_vertices;
    std::vector<Color> mesh_colors;
    std::vector<Normal> mesh_normals;
    std::vector<Triangle> mesh_polygons;
};

};

#endif