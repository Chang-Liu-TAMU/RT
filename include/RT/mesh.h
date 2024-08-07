#pragma once
#include <string>
#include <vector>
#include <memory>
#include <climits>
#include <fstream>
#include <sstream>

#include <glad/glad.h> // holds all OpenGL type declarations
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "shader.h"
#include "material.h"
#include "triangle.h"
#include "program.h"
#include "texture.h"

inline std::ostream& operator<<(std::ostream& _i, glm::vec3& _v) {
    _i << "glm::vec3 => " << "(" << _v.x << ", " << _v.y << ", " << _v.z << ")";
    return _i;
}
//
inline std::ostream& operator<<(std::ostream& _i, glm::vec4& _v) {
    _i << _v.x << " " << _v.y << " " << _v.z << " " << _v.w;
    return _i;
}

inline std::ostream& operator<<(std::ostream& _o, glm::mat4& _v) {
    for (int i = 0; i < 4; i++) {
        _o << _v[i] << " ";
    }
    return _o;
}

class Mesh;
using MESH_PTR = std::shared_ptr<Mesh>;
#define PI 3.1415926
#define MAX_BONE_INFLUENCE 4

struct Vertex {
    // position
    glm::vec3 Position;
    // normal
    glm::vec3 Normal;
    // texCoords
    glm::vec2 TexCoords;
    // tangent
    glm::vec3 Tangent;
    // bitangent
    glm::vec3 Bitangent;
    //bone indexes which will influence this vertex
    int m_BoneIDs[MAX_BONE_INFLUENCE];
    //weights from each bone
    float m_Weights[MAX_BONE_INFLUENCE];
};

#define ONES glm::mat4(1.0)

class Mesh: public std::enable_shared_from_this<Mesh> {
public:
    // mesh Data
    std::vector<Vertex>       vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture>      textures;
    std::vector<Triangle>     m_vecTriangles;
    std::string m_strName = "None";
    std::shared_ptr<Program> m_spProgram = nullptr;
    Program* m_pProgram = nullptr;
    int matType = 0;
    int texIdx = -1;

    MESH_PTR m_ptrTopMesh = nullptr;
    glm::vec3 m_vec3Position;
    bool m_bSetPosition = false;
    glm::vec3 m_vec3Ambient = glm::vec3(0.2, 0.3, 0.3);
    glm::vec3 m_vec3Diffuse = glm::vec3(0.2, 0.3, 0.5);
    glm::vec3 m_vec3Specular = glm::vec3(.2, .2, .2);
    float m_fShininess = 64.0f;

    float m_fTranslationX = 0.0;
    float m_fTranslationY = 0.0;
    float m_fTranslationZ = 0.0;
    glm::mat4 m_mat4Rotation = ONES;
    float m_fScale = -1.0;
    float m_fRotation = 0.0;
    glm::mat4 m_mat4Model = ONES;
    bool m_bChanged = false;
    std::string m_strCacheRoot = ".\\cache";

    bool m_bDisplay = true;
    bool m_bisSphere = false;

    bool m_bOutline = false;
    Program* m_pOutlineShader = nullptr;
    float m_fOutlineFactor = 1.2;
    glm::vec3 m_vec3OutlineColor = glm::vec3(1.0);

    void setOutlineShader(Program* _p) {
        m_pOutlineShader = _p;
    }

    bool needOutline() {
        if (m_ptrTopMesh) {
            return m_ptrTopMesh->needOutline();
        }
        else {
            return m_bOutline;
        }
    }

    decltype(matType) getMatType() {
        if (m_ptrTopMesh) {
            return m_ptrTopMesh->matType;
        }
        return matType;
    }

    void setTopMesh(MESH_PTR _ptr) {
        m_ptrTopMesh = _ptr;
    }

    void resetTopMesh() {
        m_ptrTopMesh = nullptr;
    }
    
    void outlineOn(){
        m_bOutline = true;
    }

    void outlineOff() {
        m_bOutline = false;
    }

    void setIsSphere(bool _b = true) {
        m_bisSphere = _b;
    }

    bool isSphere() {
        if (m_ptrTopMesh) {
            return m_ptrTopMesh->isSphere();
        }
        return m_bisSphere;
    }

    void display() {
        m_bDisplay = true;
        std::cout << "INFO: model " << m_strName << " is displayed.\n";
    }

    void undisplay() {
        m_bDisplay = false;
        std::cout << "INFO: model " << m_strName << " is undisplayed.\n";
    }

    Texture *getFirstTexture() {
        if (textures.empty()) {
            return nullptr;
        }
        return &textures[0];
    }

    virtual void shareMeshData(MESH_PTR _p) {
        int base = _p->vertices.size();
        for (auto v : vertices) {
            _p->vertices.push_back(v);
        }
        for (auto i : indices) {
            _p->indices.push_back(i + base);
        }
        if (_p->textures.size() >= 1) {
            return;
        }
        for (auto t : textures) {
            _p->textures.push_back(t);
        }
    }

#define WRITE(X, Y) X << #Y << " " << Y << std::endl  
    void positionSnapshot() {
        std::ofstream ofs(m_strCacheRoot + "\\position\\" + m_strName + ".txt");
        WRITE(ofs, m_fTranslationX);
        WRITE(ofs, m_fTranslationY);
        WRITE(ofs, m_fTranslationZ);
        WRITE(ofs, m_fScale);
        WRITE(ofs, m_mat4Rotation);
        std::cout << "INFO: save position of mesh " << m_strName << std::endl;
    }

#define READIN(M, K) K = M[#K][0]
    void loadSnapshot() {
        std::ifstream ifs(m_strCacheRoot + "\\position\\" + m_strName + ".txt");
        if (ifs.fail()) {
            std::cout << "INFO: Fail to load position for mesh " << m_strName << std::endl;
            return;
        }
        std::string line;
        std::vector<std::string> cache;
        std::map<std::string, std::vector<float>> posInfo;
        while (std::getline(ifs, line)) {
            cache.clear();
            std::istringstream iss(line);
            std::string tem;
            while (iss >> tem) {
                cache.push_back(tem);
            }
            auto& ref = posInfo[cache[0]];
            for (int i = 1; i < cache.size(); i++) {
                ref.push_back(std::stof(cache[i]));
            }
        }
        if (posInfo.empty()) {
            std::cout << "INFO: Position is empty for mesh " << m_strName << std::endl;
            return;
        }
        READIN(posInfo, m_fTranslationX);
        READIN(posInfo, m_fTranslationY);
        READIN(posInfo, m_fTranslationZ);
        READIN(posInfo, m_fScale);
        m_mat4Rotation = glm::make_mat4x4(posInfo["m_mat4Rotation"].data());
        std::cout << "INFO: load position of mesh " << m_strName << std::endl;
    }

    void setName(std::string _name) noexcept {
        m_strName = _name;
    }

    std::string getName() const noexcept  {
        return m_strName;
    }
    
    void registerThisModel(std::map<std::string, MESH_PTR>& _m) {
        if (_m.find(m_strName) == _m.end()) {
            _m[m_strName] = shared_from_this();
        }
        else {
            std::cout << "WARNING: mesh named with " << m_strName << " already exists in table. This model'll be ignored.\n";
        }
    }

    virtual void genTriangles(std::vector<Triangle>& _v, bool _setNormal = false);

    virtual glm::vec3 resolvePosition() {
        m_vec3Position = glm::vec3(0.0, 0.0, 0.0);
        for (auto& v : vertices) {
            m_vec3Position += v.Position;
        }
        m_vec3Position /= vertices.size();
        m_bSetPosition = true;
        return m_vec3Position;
    }

    float getAverageRadius() {
        float scale = m_fScale;
        if (m_ptrTopMesh) {
            scale = m_ptrTopMesh->m_fScale;
        }
        else {
            std::cout << m_strName << std::endl;
        }
        float x = 1.7320508075688772;
        auto r = glm::distance(m_vec3Position, vertices[0].Position) * scale / x;
        //std::cout << r << std::endl;
        return r;
    }

    glm::vec3 getPosition() {
        if (!m_bSetPosition) {
            resolvePosition();
        }
        return glm::vec3(getModel() * glm::vec4(m_vec3Position, 1.0));
    }

    void setMatType(int _m) { 
        matType = _m;
    }

    glm::mat4 getModel(bool _outline = false) {
        if (m_ptrTopMesh) {
            return m_ptrTopMesh->getModel(_outline);
        }

        float outlineFactor = _outline ? m_fOutlineFactor : 1.0;
        auto m = ONES;
        m = glm::translate(m, glm::vec3(m_fTranslationX, m_fTranslationY, m_fTranslationZ));
        m *= m_mat4Rotation;
        auto scale = m_fScale * outlineFactor;
        m = glm::scale(m, glm::vec3(scale, scale, scale));
        m_mat4Model = m;
        return m;
    }

    void scaleCommand(float _amp) {
        m_fScale *= _amp;
    }

    float getRadians(float _d) {
        return _d / 180.f * PI;
    }

    void rotationXCommand(float _d) {
        m_mat4Rotation = glm::rotate(ONES, getRadians(_d), glm::vec3(1.0, 0.0, 0.0)) * m_mat4Rotation;
    }

    void rotationYCommand(float _d) {
        m_mat4Rotation = glm::rotate(ONES, getRadians(_d), glm::vec3(0.0, 1.0, 0.0)) * m_mat4Rotation;
    }

    void rotationZCommand(float _d) {
        m_mat4Rotation = glm::rotate(ONES, getRadians(_d), glm::vec3(0.0, 0.0, 1.0)) * m_mat4Rotation;
    }

    void translateXCommand(float _d) {
        m_fTranslationX += _d;
    }

    void translateYCommand(float _d) {
        m_fTranslationY += _d;
    }

    void translateZCommand(float _d) {
        m_fTranslationZ += _d;
    }

    void translateCommand(float _x, float _y, float _z) {
        m_fTranslationX += _x;
        m_fTranslationY += _y;
        m_fTranslationZ += _z;
    }

    void translateCommand(vec3 _v) {
        translateCommand(_v[0], _v[1], _v[2]);
    }

    void translateCommand(glm::vec3 _v) {
        translateCommand(_v[0], _v[1], _v[2]);
    }

    void ambientCommand(float _r, float _g, float _b) {
        m_vec3Ambient.r = _r;
        m_vec3Ambient.g = _g;
        m_vec3Ambient.b = _b;
    }

    void diffuseCommand(float _r, float _g, float _b) {
        m_vec3Diffuse.r = _r;
        m_vec3Diffuse.g = _g;
        m_vec3Diffuse.b = _b;
    }

    void specularCommand(float _r, float _g, float _b) {
        m_vec3Specular.r = _r;
        m_vec3Specular.g = _g;
        m_vec3Specular.b = _b;
    }

    void resetAll() {
        m_mat4Rotation = ONES;
        m_fTranslationX = 0.0;
        m_fTranslationY = 0.0;
        m_fTranslationZ = 0.0;
        m_fScale = 1.0;
        m_fRotation = 0.0;
    }

    void setTexIdx(unsigned int _texIdx) {
        texIdx = _texIdx;
    }

    unsigned int VAO = UINT_MAX;
    GLenum error;

    // constructor
    Mesh() = default;
    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<Texture> textures, 
        int _matType = 0, std::string _name= "None");

    virtual ~Mesh() {
        /*if (VAO != UINT_MAX) {
            glDeleteVertexArrays(1, &VAO);
        }
        if (VBO != UINT_MAX) {
            glDeleteBuffers(1, &VBO);
        }
        if (EBO != UINT_MAX) {
            glDeleteBuffers(1, &EBO);
        }*/
    }

    void checkGlError(const char * _m) {
        error = glGetError();
        while (error != GL_NO_ERROR) {
            std::cout << "GL Error code: " << error << " " << _m << std::endl;
            error = glGetError();
        }
    }

    void setPositionFromVec3(vec3& _v) {
        m_fTranslationX = _v[0];
        m_fTranslationY = _v[1];
        m_fTranslationZ = _v[2];
    }

    // render the mesh
    void virtual Draw(glm::mat4& _proj, glm::mat4& _view, Mesh* _p);

    void setDiffuse(glm::vec3 &v) {
        m_vec3Diffuse = v;
    }

    void setSpecular(glm::vec3& v) {
        m_vec3Specular = v;
    }

    void setShininess(float _f) {
        m_fShininess = _f;
    }

    void setProgram(std::shared_ptr<Program> _p) {
        m_spProgram = _p;
    }

    void setProgram(Program *_p) {
        m_pProgram = _p;
    }

    virtual void setProgramSubMesh(Program* _p) {
        return;
    }

    virtual void setMatTypeSubMesh(int _m) {
        return;
    }
    // render data 
    unsigned int VBO = UINT_MAX, EBO = UINT_MAX;

    void setVAOattributes();

    // initializes all the buffer objects/arrays
    void virtual setupMesh();

    void setupMesh2();
};

class LightMesh : public Mesh {
public:
    using Mesh::Mesh;
    ~LightMesh() override {};
    void Draw(glm::mat4& _proj, glm::mat4& _view, Mesh* _p) override;    
    
    void setupMesh() override;
};

#define GEN_SET_FUNC(X) void set##X##Face(MESH_PTR _p){ m_ptr##X##Face = _p; _p->m_ptrTopMesh = shared_from_this(); } \
        MESH_PTR get##X##Face(){ return m_ptr##X##Face; }
#define DRAW_FACE(X) m_ptr##X##Face->Draw(_proj, _view, _p)
class CubeMesh : public Mesh {
public:
    CubeMesh(MESH_PTR _front, MESH_PTR _back, MESH_PTR _left, MESH_PTR _right,
            MESH_PTR _up, MESH_PTR _bottom) : Mesh::Mesh({}, {}, {}) {
        m_ptrFrontFace = _front;
        m_ptrBackFace = _back;
        m_ptrLeftFace = _left;
        m_ptrRightFace = _right;
        m_ptrUpFace = _up;
        m_ptrBottomFace = _bottom;
    }

    ~CubeMesh() override {};
    void Draw(glm::mat4& _proj, glm::mat4& _view, Mesh* _p) override {
        DRAW_FACE(Front);
        DRAW_FACE(Back);
        DRAW_FACE(Left);
        DRAW_FACE(Right);
        DRAW_FACE(Up);
        DRAW_FACE(Bottom);
    };

    void setupMesh() override {
        return;
    };

    GEN_SET_FUNC(Front);
    GEN_SET_FUNC(Back);
    GEN_SET_FUNC(Left);
    GEN_SET_FUNC(Right);
    GEN_SET_FUNC(Up);
    GEN_SET_FUNC(Bottom);

#define APPLY_TO_ALL(F, X) m_ptrFrontFace->F(X);\
    m_ptrBackFace->F(X); \
    m_ptrLeftFace->F(X); \
    m_ptrRightFace->F(X); \
    m_ptrUpFace->F(X); \
    m_ptrBottomFace->F(X); \

    void setProgramSubMesh(Program* _p) override {
        APPLY_TO_ALL(setProgram, _p);
    }

    void setMatTypeSubMesh(int _m) override {
        APPLY_TO_ALL(setMatType, _m);
    }

    void genTriangles(std::vector<Triangle>& _v, bool _setNormal = false) override {
        APPLY_TO_ALL(genTriangles, _v)
    }

    glm::vec3 resolvePosition() override {
        m_vec3Position = glm::vec3(0.0, 0.0, 0.0);
        m_vec3Position += m_ptrFrontFace->resolvePosition();
        m_vec3Position += m_ptrBackFace->resolvePosition();
        m_vec3Position += m_ptrLeftFace->resolvePosition();
        m_vec3Position += m_ptrRightFace->resolvePosition();
        m_vec3Position += m_ptrUpFace->resolvePosition();
        m_vec3Position += m_ptrBottomFace->resolvePosition();
        m_vec3Position /= 6.0;
        m_bSetPosition = true;
        return m_vec3Position;
    }

    void shareMeshData(MESH_PTR _p) override {
        APPLY_TO_ALL(shareMeshData, _p);
    }

private:
    MESH_PTR m_ptrLeftFace;
    MESH_PTR m_ptrRightFace;
    MESH_PTR m_ptrFrontFace;
    MESH_PTR m_ptrBackFace;
    MESH_PTR m_ptrUpFace;
    MESH_PTR m_ptrBottomFace;
};



class MeshList : public Mesh {
public:
    MeshList() : Mesh::Mesh({}, {}, {}) {
    }

    ~MeshList() override {};

    void Draw(glm::mat4& _proj, glm::mat4& _view, Mesh* _p) override {
        for (auto& p : m_vecMeshes) {
            p->Draw(_proj, _view, _p);
        }
    };

    void setupMesh() override {
        return;
    };

    void addMesh(MESH_PTR _p) {
        m_vecMeshes.push_back(_p);
        _p->m_ptrTopMesh = shared_from_this();
    }

    void setProgramSubMesh(Program* _p) override {
        for (auto& p : m_vecMeshes) {
            p->setProgram(_p);
        }
    }

    void setMatTypeSubMesh(int _m) override {
        for (auto& p : m_vecMeshes) {
            p->setMatType(_m);
        }
    }

    void genTriangles(std::vector<Triangle>& _v, bool _setNormal = false) override {
        for (auto& p : m_vecMeshes) {
            p->genTriangles(_v);
        }
    }

    glm::vec3 resolvePosition() override {
        m_vec3Position = glm::vec3(0.0, 0.0, 0.0);
        for (auto& p : m_vecMeshes) {
            m_vec3Position += p->resolvePosition();
        }
        m_vec3Position /= m_vecMeshes.size();
        m_bSetPosition = true;
        return m_vec3Position;
    }

private:
    std::vector<MESH_PTR> m_vecMeshes;
};

class FlyWeight : public Mesh {
public:
    FlyWeight(MESH_PTR _ptr) : _orig(_ptr) {}

    FlyWeight() : Mesh::Mesh({}, {}, {}) {
    }

    ~FlyWeight() override {};

    void Draw(glm::mat4& _proj, glm::mat4& _view, Mesh* _p) override {
        _orig->setTopMesh(shared_from_this());
        _orig->Draw(_proj, _view, _p);
        _orig->resetTopMesh();
    };

    void setupMesh() override {
        return;
    };

    void setMatTypeSubMesh(int _m) override {
         _orig->setMatType(_m);
    }

    void genTriangles(std::vector<Triangle>& _v, bool _setNormal = false) override {
        _orig->setTopMesh(shared_from_this());
        _orig->genTriangles(_v);
        _orig->resetTopMesh();
    }

private:
    MESH_PTR _orig;
};

