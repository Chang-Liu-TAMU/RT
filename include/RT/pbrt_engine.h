#pragma once
#include <memory>
#include <iostream>
#include <algorithm>
#include <mutex>

#include "stdlib.h"
#include "vec3.h"
#include "ray.h"
#include "material.h"
#include "triangle.h"
#include "rt_camera.h"
#include "rt_model.h"
#include "camera.h"
#include "texture.h"
#include "mesh.h"

using TRI_PTR = std::shared_ptr<Triangle[]>;
using BVH_PTR = std::shared_ptr<bvh_node[]>;
using VEC_TRI = std::vector<Triangle>;
#define STACK_SIZE 25
#define RET_STACK_SIZE 20
#define FLAG_STACK_SIZE 25
#define NX (1280 / 2)
#define NY (720 / 2)
#define NS (100 / 10)

#define MAT_NUM 50
class Window;
__global__ void rand_init(curandState* rand_state);

void make_bvh_nodes(std::vector<PtrIdxPair>& _p, int m, int n, float time0, float time1, std::vector<bvh_node>& vecBvh);

__global__ void camera_init(camera** cam, int nx, int ny, vec3 lookfrom, vec3 lookat, float vfov, 
	float dist_to_focus = 10.0, float aperture = 0.0);

__global__ void camera_update(camera** cam, int nx, int ny, vec3 lookfrom, vec3 lookat, float vfov, 
	float dist_to_focus = 10.0, float aperture = 0.0);

__global__ void material_init(material** mat, curandState* _s);

__device__ bool hit(const ray& r, const bvh_node* world, int bvh_num, int tri_num, bool* flags, int* stack, hit_record* retStack, const Triangle* tri, float t_min, float t_max, hit_record& rec);

__device__ vec3 color(const ray& r, const bvh_node* world, int bvh_num, int tri_num, const Triangle* tri, curandState* s,
	bool* visitedFlags, int* stack, hit_record* retStack, material** mats, TEX_OBJ _texObj, cudaTextureObject_t _cubeMap);

__global__ void render(vec3* fb, int max_x, int max_y, int ns, const bvh_node* bvh, int bvh_num, int tri_num, Triangle* tri, camera** cam, curandState* randState,
	bool* h_visitedFlags, int* h_stack, hit_record* h_retStack, material** mat, TEX_OBJ _texObj, cudaTextureObject_t _cubeMap);

class Mesh;
class PbrtEngine {
public:
	static PbrtEngine& getInstance() {
		static PbrtEngine* engine = new PbrtEngine;
		return *engine;
	}

	PbrtEngine(int nx = NX, int ny = NY, int ns = NS);

	~PbrtEngine() {
		__releaseMemory();
	}
	void renderScene(bool _buildScene=true);

	void setNx(int _nx) {
		if (_nx != nx) nx_ny_changed = true;
		nx = _nx;
	}

	void setNy(int _ny) {
		if (_ny != ny) nx_ny_changed = true;
		ny = _ny;
	}

	void setNs(int _ns) {
		ns = _ns;
	}

	void refreshPixelNum();

	void uponPixelNumChange() {
		refreshPixelNum();
		__allocateCudaStackMemory();
	}


	void set(int _nx, int _ny, int _ns) {
		nx = _nx;
		ny = _ny;
		ns = _ns;
		num_pixels = nx * ny;
	}

	bool nXnYChanged() {
		auto tem = nx_ny_changed;
		nx_ny_changed = false;
		return tem;
	}

	void addMesh(std::vector<Mesh*> _m);
	void addMesh(Mesh* _m);

	void saveBvhNodesToFile();

	void genSomeMaterials();

	std::vector<Mesh*> m_vecScene;

	std::vector<Triangle> m_vecTriangles;

	int nx, ny, ns, num_pixels, tx = 16, ty = 16;

	size_t fb_size;

	vec3* fb = nullptr;
	vec3* h_fb = nullptr;

	curandState* d_rand_state = nullptr;
	curandState* d_rand_state2 = nullptr;

	int vecBvhNodesNum;

	bool nx_ny_changed = false;

	void checkStatus() {
		if (worker.joinable()) {
			std::cout << "PbrtEngine: rendering not finished.\n";
		}
		else {
			std::cout << "PbrtEngine: rendering finished.\n";
		}
	}

	void genTextures() {
		d_texObjs = cuManyTextures(m_iTexObjsNum);
		return;
		int texNum = m_vecScene.size();
		int i = 0;
		for (auto* ptr : m_vecScene) {
			auto* texPtr = ptr->getFirstTexture();
			if (!texPtr) {
				std::cout << "Fail: texPtr is nullptr.\n";
				continue;
			}
			std::string name = texPtr->name;
			std::string path = texPtr->path;
			ptr->setTexIdx(0);
			d_texObjs = cuManyTextures(m_iTexObjsNum);
			break;
		} 

		cubeMapObj = cuTextureCubeMapFromFile();
	}

	void join();

private:

	void __renderScene(bool _genScene);
	void __genTrianglesFromScene();
	void __genBvhNodes();
	void __allocateCudaStackMemory();

#define CUFREE(X) if (X) { cudaFree(X); X = nullptr;}
	void __releaseMemory() {
		CUFREE(d_bvh_nodes);
		CUFREE(d_triangles);
		CUFREE(cam);
		CUFREE(d_mats);
		CUFREE(h_visitedFlags);
		CUFREE(h_stack);
		CUFREE(h_retStack);
		CUFREE(d_rand_state);
		CUFREE(d_rand_state2);
		CUFREE(d_bvh_nodes);
		CUFREE(d_triangles);
		CUFREE(fb);
		CUFREE(d_texObjs);
		if (h_fb) {
			delete h_fb;
			h_fb = nullptr;
		}
	}

	bvh_node* d_bvh_nodes = nullptr;
	Triangle* d_triangles = nullptr;
	camera** cam = nullptr;

	material** d_mats = nullptr;
	bool* h_visitedFlags = nullptr;
	int* h_stack = nullptr;

	hit_record* h_retStack = nullptr;

	std::string m_strRenderStatus = "None";

	std::thread worker;

	std::mutex mtx;
	
	TEX_OBJ d_texObjs;
	cudaTextureObject_t cubeMapObj;
	int m_iTexObjsNum;
};