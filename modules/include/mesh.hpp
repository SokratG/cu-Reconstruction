#ifndef CUPHOTO_LIB_MESH_HPP
#define CUPHOTO_LIB_MESH_HPP

#include "types.hpp"

#include <vector>
#include <string>

namespace cuphoto {


class Mesh {
public:
    using Point = Vec3f;
    using ColorPoint = Vec6f;
    using NormalColorPoint = Vec10f;
    using Triangle = Vec3i;
public:
    Mesh() = default;

    std::vector<Mesh::NormalColorPoint> vertices() const;

    void vertices(const std::vector<Mesh::NormalColorPoint>& mesh_vertices);

    std::vector<Mesh::Triangle> polygons() const;

    void polygons(const std::vector<Mesh::Triangle>& mesh_faces);

    void store_to_ply(const std::string& ply_filepath) const;

private:
    std::vector<NormalColorPoint> mesh_vertices;
    std::vector<Triangle> mesh_polygons;
};

};

#endif