#include "mesh.hpp"

namespace cuphoto {

std::vector<r32> Mesh::vertices() const {
    return mesh_vertices;
}


void Mesh::vertices(const std::vector<r32>& _mesh_vertices) {
    mesh_vertices = _mesh_vertices;
}

std::vector<i32> Mesh::faces() const {
    return mesh_faces;
}

void Mesh::faces(const std::vector<i32>& _mesh_faces) {
    mesh_faces = _mesh_faces;
}

};