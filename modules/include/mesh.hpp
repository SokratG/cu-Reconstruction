#ifndef CUPHOTO_LIB_MESH_HPP
#define CUPHOTO_LIB_MESH_HPP

#include "types.hpp"

#include <vector>

namespace cuphoto {


class Mesh {
public:
    Mesh() = default;

    std::vector<r32> vertices() const;

    void vertices(const std::vector<r32>& mesh_vertices);

    std::vector<i32> faces() const;

    void faces(const std::vector<i32>& mesh_faces);

private:
    std::vector<r32> mesh_vertices;
    std::vector<i32> mesh_faces;
};

};

#endif