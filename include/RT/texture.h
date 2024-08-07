#pragma once
#include "common.h"
#define TEX_OBJ cudaTextureObject_t*
//#define TEX_OBJ CudaTexture

struct Texture;
static std::map<std::string, Texture> textureDict;
struct Texture {
    Texture() {}
    Texture(std::string _name, std::string _type = "diffuse");
    static Texture getTexture(std::string _name, std::string _type="diffuse") {
        auto it = textureDict.try_emplace(_name, _name, _type);
        return it.first->second;
    }
    unsigned int id;
    std::string type;
    std::string path = "./images";
    std::string name = "None";

    std::string getPath() {
        return path + "/" + name;
    }
};

struct RGB {
    float r;
    float g;
    float b;
    float a;
};

struct CudaTexture {
    __device__ __host__ CudaTexture(float* _data, int _w, int _h) {
        d_data = _data;
        width = _w;
        height = _h;
    }
    __device__ __host__ CudaTexture() = default;

    float* d_data = nullptr;
    int width = 0;
    int height = 0;

    __device__ __host__ RGB fetch(float _u, float _v) {
        int u = (width - 1) * _u;
        int v = (height - 1) * _v;
        RGB ret;
        //int base = u * width * 4 * sizeof(float) + v * 4 * sizeof(float); //no sizeof(float) !!! this is a bug
        int base = u * width * 4 + v * 4; //no sizeof(float) fixed
        ret.r = d_data[base];
        ret.g = d_data[base + 1];
        ret.b = d_data[base + 2];
        return ret;
    }

};
cudaTextureObject_t* cuManyTextures(int& _num);

CudaTexture cuTextureFromFile(const char* name, const std::string& directory);

unsigned int TextureFromFile(const char* path, const std::string& directory, bool gamma=false);

cudaTextureObject_t cuTextureFromFile(const char* path, const std::string& directory, bool gamma);

cudaTextureObject_t cuTextureCubeMapFromFile(const char* _dir = "./images/cubemap", bool gamma = false);
