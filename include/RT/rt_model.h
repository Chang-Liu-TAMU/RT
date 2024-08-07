#pragma once
#ifndef MODEL_H
#define MODEL_H

#include "common.h"
#include <glad/glad.h> 

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <vector>

#include "triangle.h"
#include "vec3.h"
using namespace std;

#pragma once
#ifndef MESH_H
#define MESH_H

#include <glad/glad.h> // holds all OpenGL type declarations

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>"

#include <string>
#include <vector>
using namespace std;

struct RtVertex {
    // position
    glm::vec3 Position;
    // normal
    glm::vec3 Normal;
    // texCoords
    glm::vec2 TexCoords;
};


class RtMesh {
public:
    // mesh Data
    vector<RtVertex>       vertices;
    vector<unsigned int> indices;
    // constructor
    RtMesh(vector<RtVertex> vertices, vector<unsigned int> indices)
    {
        this->vertices = vertices;
        this->indices = indices;
        // now that we have all the required data, set the vertex buffers and its attribute pointers.
    }

};
#endif

class RtModel
{
public:
    string directory;
    float scale = 1.0;
    vec3 transition;
    int matType = 0;
    // constructor, expects a filepath to a 3D model.
    RtModel(string const& path, int _matType = 0, float _s = 1.0, vec3 _t = vec3(0.0, 0.0, 0.0)): matType(_matType), scale(_s), transition(_t)
    {
        loadModel(path);
    }

    std::vector<RtMesh> meshes;
    int triNum = 0;
    std::vector<Triangle> genTriangleVector() {
        std::vector<Triangle> ret;
        if (meshes.empty()) {
            std::cout << "Error: Mesh is empty.\n";
        }
        auto& firstMesh = meshes[0];
        auto& ref1 = firstMesh.indices;
        triNum = ref1.size() / 3;
        for (int i = 0; i < triNum; i++) {
            auto v1 = cvtGlmVec3ToVec3(firstMesh.vertices[ref1[i*3]].Position);
            auto v2 = cvtGlmVec3ToVec3(firstMesh.vertices[ref1[i*3+1]].Position);
            auto v3 = cvtGlmVec3ToVec3(firstMesh.vertices[ref1[i*3+2]].Position);
            ret.push_back(Triangle(v1 * scale + transition, v2 * scale + transition, 
                v3 * scale + transition, matType));
        }
        return ret;
    }

private:
    // loads a model with supported ASSIMP extensions from file and stores the resulting meshes in the meshes vector.
    void loadModel(string const& path)
    {
        // read file via ASSIMP
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
        // check for errors
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
        {
            cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << endl;
            return;
        }
        // retrieve the directory path of the filepath
        directory = path.substr(0, path.find_last_of('/'));

        // process ASSIMP's root node recursively
        processNode(scene->mRootNode, scene);
    }

    // processes a node in a recursive fashion. Processes each individual mesh located at the node and repeats this process on its children nodes (if any).
    void processNode(aiNode* node, const aiScene* scene)
    {
        // process each mesh located at the current node
        for (unsigned int i = 0; i < node->mNumMeshes; i++)
        {
            // the node object only contains indices to index the actual objects in the scene. 
            // the scene contains all the data, node is just to keep stuff organized (like relations between nodes).
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            meshes.push_back(processMesh(mesh, scene));
        }
        // after we've processed all of the meshes (if any) we then recursively process each of the children nodes
        for (unsigned int i = 0; i < node->mNumChildren; i++)
        {
            processNode(node->mChildren[i], scene);
        }

    }

    RtMesh processMesh(aiMesh* mesh, const aiScene* scene)
    {
        // data to fill
        vector<RtVertex> vertices;
        vector<unsigned int> indices;

        // walk through each of the mesh's vertices
        for (unsigned int i = 0; i < mesh->mNumVertices; i++)
        {
            RtVertex vertex;
            glm::vec3 vector; // we declare a placeholder vector since assimp uses its own vector class that doesn't directly convert to glm's vec3 class so we transfer the data to this placeholder glm::vec3 first.
            // positions
            vector.x = mesh->mVertices[i].x;
            vector.y = mesh->mVertices[i].y;
            vector.z = mesh->mVertices[i].z;
            vertex.Position = vector;
            // normals
            if (mesh->HasNormals())
            {
                vector.x = mesh->mNormals[i].x;
                vector.y = mesh->mNormals[i].y;
                vector.z = mesh->mNormals[i].z;
                vertex.Normal = vector;
            }
            // texture coordinates
            if (mesh->mTextureCoords[0]) // does the mesh contain texture coordinates?
            {
                glm::vec2 vec;
                // a vertex can contain up to 8 different texture coordinates. We thus make the assumption that we won't 
                // use models where a vertex can have multiple texture coordinates so we always take the first set (0).
                vec.x = mesh->mTextureCoords[0][i].x;
                vec.y = mesh->mTextureCoords[0][i].y;
                vertex.TexCoords = vec;
                // tangent
                vector.x = mesh->mTangents[i].x;
                vector.y = mesh->mTangents[i].y;
                vector.z = mesh->mTangents[i].z;
                // bitangent
                vector.x = mesh->mBitangents[i].x;
                vector.y = mesh->mBitangents[i].y;
                vector.z = mesh->mBitangents[i].z;
            }
            else {
                vertex.TexCoords = glm::vec2(0.0f, 0.0f);
            }

            vertices.push_back(vertex);
        }
        // now wak through each of the mesh's faces (a face is a mesh its triangle) and retrieve the corresponding vertex indices.
        for (unsigned int i = 0; i < mesh->mNumFaces; i++)
        {
            aiFace face = mesh->mFaces[i];
            // retrieve all indices of the face and store them in the indices vector
            for (unsigned int j = 0; j < face.mNumIndices; j++)
                indices.push_back(face.mIndices[j]);
        }

        return RtMesh(vertices, indices);
    }


};

#endif