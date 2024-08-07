#pragma once
#include <string>
#include <vector>
#include <time.h>
#include <stdlib.h>

#include <glad/glad.h> // holds all OpenGL type declarations
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "shader.h"
#include "mesh.h"
#include "camera.h"

Mesh::Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<Texture> textures, 
    int _m, std::string _name): matType(_m), m_strName(_name)
{
    this->vertices = std::move(vertices);
    this->indices = std::move(indices);
    this->textures = std::move(textures);
}

// render the mesh
void Mesh::Draw(glm::mat4& _proj, glm::mat4& _view, Mesh* _p)
{
    if (!m_bDisplay) {
        return;
    }

    checkGlError("draw\n");
    auto* shader = m_pProgram;
    shader->use();
    // bind appropriate textures
    unsigned int diffuseNr = 1;
    unsigned int specularNr = 1;
    unsigned int normalNr = 1;
    unsigned int heightNr = 1;
    

    for (unsigned int i = 0; i < textures.size(); i++)
    {
        glActiveTexture(GL_TEXTURE0 + i); // active proper texture unit before binding
        // retrieve texture number (the N in diffuse_textureN)
        std::string number;
        std::string name = textures[i].type;
        if (name == "texture_diffuse")
            number = std::to_string(diffuseNr++);
        else if (name == "texture_specular")
            number = std::to_string(specularNr++); // transfer unsigned int to string
        else if (name == "texture_normal")
            number = std::to_string(normalNr++); // transfer unsigned int to string
        else if (name == "texture_height")
            number = std::to_string(heightNr++); // transfer unsigned int to string

        // now set the sampler to the correct texture unit
        shader->setInt((name + number).c_str(), i);
        // and finally bind the texture
        glBindTexture(GL_TEXTURE_2D, textures[i].id);
    }

    auto* pLight = static_cast<LightMesh*>(_p); //difference between static_cast and dynamic_cast
    
    //auto model = glm::mat4(1.0f);
    shader->setMat4("model", getModel());
    shader->setMat4("view", _view);
    shader->setMat4("projection", _proj);
    
    // light properties
    shader->setVec3("light.position", pLight->m_vec3Position);
    shader->setVec3("light.ambient", pLight->m_vec3Ambient);
    shader->setVec3("light.diffuse", pLight->m_vec3Diffuse);
    shader->setVec3("light.specular", pLight->m_vec3Specular);
    
    // material properties
    shader->setVec3("material.diffuse", m_vec3Diffuse);
    shader->setVec3("material.specular", m_vec3Specular);
    shader->setFloat("material.shininess", m_fShininess);

    
    shader->setVec3("viewPos", Camera::getInstance().m_v3Position);
    
    if (m_bOutline && m_pOutlineShader) {
        glStencilFunc(GL_ALWAYS, 1, 0xff);
        glStencilMask(0xff);
    }

    checkGlError("after before shader\n");
    // draw mesh
    checkGlError("before VAO\n");
    glBindVertexArray(VAO);
    checkGlError("before draw elements\n");
    glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);

    if (m_bOutline && m_pOutlineShader) {
        m_pOutlineShader->use();
        m_pOutlineShader->setMat4("model", getModel(true));
        m_pOutlineShader->setMat4("view", _view);
        m_pOutlineShader->setMat4("projection", _proj);
        m_pOutlineShader->setVec3("outlineColor", m_vec3OutlineColor);

        glStencilFunc(GL_NOTEQUAL, 1, 0xff);
        glStencilMask(0x00);
        glDisable(GL_DEPTH_TEST);

        glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);
        glEnable(GL_DEPTH_TEST);
        //glStencilMask(0xff);
        //glDisable(GL_STENCIL_TEST);
    }

    glBindVertexArray(0);
    checkGlError("after draw elements\n");
    // always good practice to set everything back to defaults once configured.
    glActiveTexture(GL_TEXTURE0);

    
}

void Mesh::genTriangles(std::vector<Triangle> & _v, bool _setNormal) {
    if (m_strName == "girl") {
        _setNormal = true;
    }
    if (isSphere()) {
        //todo only need to generate a single triangle
        auto pos = getPosition();
        auto r = getAverageRadius();
        _v.push_back(Triangle(cvtGlmVec3ToVec3(pos), r, getMatType()));
        return;
    }

    auto& firstMesh = *this;
    auto& ref1 = firstMesh.indices;
    int triNum = ref1.size() / 3;
    auto model = getModel();

    for (int i = 0; i < triNum; i++) {
        //todo apply trans to each vertex only once
        auto& p1 = firstMesh.vertices[ref1[i * 3]];
        auto v1 = glm::vec3(model * glm::vec4(p1.Position, 1.0));
        auto uv1 = p1.TexCoords;
        auto& p2 = firstMesh.vertices[ref1[i * 3 + 1]];
        auto v2 = glm::vec3(model * glm::vec4(p2.Position, 1.0));
        auto uv2 = p2.TexCoords;
        auto& p3 = firstMesh.vertices[ref1[i * 3 + 2]];
        auto v3 = glm::vec3(model * glm::vec4(p3.Position, 1.0));
        auto uv3 = p3.TexCoords;
    
        _v.push_back(Triangle(cvtGlmVec3ToVec3(v1), cvtGlmVec3ToVec3(v2), cvtGlmVec3ToVec3(v3), getMatType(), texIdx,
                                vec3(uv1[0], uv2[0], uv3[0]), vec3(uv1[1], uv2[1], uv3[1])));
        if (_setNormal) {
            auto& curr = _v.back();
            curr.setNormal(cvtGlmVec3ToVec3(p1.Normal), cvtGlmVec3ToVec3(p2.Normal), cvtGlmVec3ToVec3(p3.Normal));
        }
    }
    return;
}

void Mesh::setVAOattributes() {
    // set the vertex attribute pointers
    // vertex Positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    // vertex normals
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
    // vertex texture coords
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));
    // vertex tangent
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Tangent));
    // vertex bitangent
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Bitangent));
    // ids
    glEnableVertexAttribArray(5);
    glVertexAttribIPointer(5, 4, GL_INT, sizeof(Vertex), (void*)offsetof(Vertex, m_BoneIDs));

    // weights
    glEnableVertexAttribArray(6);
    glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, m_Weights));
    glBindVertexArray(0);
}

// initializes all the buffer objects/arrays
void Mesh::setupMesh() 
{
    // create buffers/arrays
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    // load data into vertex buffers
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // A great thing about structs is that their memory layout is sequential for all its items.
    // The effect is that we can simply pass a pointer to the struct and it translates perfectly to a glm::vec3/2 array which
    // again translates to 3/2 floats which translates to a byte array.
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

    setVAOattributes();
}

void Mesh::setupMesh2()
{
    // create buffers/arrays
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    // load data into vertex buffers
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // A great thing about structs is that their memory layout is sequential for all its items.
    // The effect is that we can simply pass a pointer to the struct and it translates perfectly to a glm::vec3/2 array which
    // again translates to 3/2 floats which translates to a byte array.
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

    setVAOattributes();
}


void LightMesh::Draw(glm::mat4& _proj, glm::mat4& _view, Mesh* _p) {  
    if (!m_bDisplay) {
        return;
    }
    Program* shader = m_pProgram;
    auto* pLight = static_cast<LightMesh*>(_p); //difference between static_cast and dynamic_cast
    shader->use();
    shader->setMat4("model", getModel());
    shader->setMat4("view", _view);
    shader->setMat4("projection", _proj);

    // light properties
    shader->setVec3("lightColor", m_vec3Diffuse);

    // draw mesh
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, static_cast<unsigned int>(indices.size()), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    // always good practice to set everything back to defaults once configured.
    glActiveTexture(GL_TEXTURE0);
    return;
}

void LightMesh::setupMesh(){
    resolvePosition();

    m_vec3Position = glm::vec3(0.0, 0.0, 0.0);
    m_vec3Ambient = glm::vec3(0.3, 0.3, 0.3);
    m_vec3Diffuse = glm::vec3(1.0, 1.0, 1.0);
    m_vec3Specular = glm::vec3(1.0, 1.0, 1.0);

    Mesh::setupMesh();
}

