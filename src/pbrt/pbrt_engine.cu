#include "thread"

#include "pbrt_engine.h"

//#define DIELECTRIC 7
//#define LAMBERTIAN 4
//#define METAL 6
//#define LIGHT 3

#define MAT1 "lambertian(vec3(0.8, 0.8, 0.9))"
#define MAT2 "metal(vec3(5.0, 5.0, 5.0f), 0.1)"
#define MAT3 "dielectric(0.4)"
#define MAT4 "diffusive_light(vec3(1.0, 1.0, 1.0))"
#define MAT5 "lambertian(vec3(1.0, 0.75, 0.95))"
#define MAT6 "lambertian(vec3(CURAND(_s), CURAND(_s), CURAND(_s)))"
#define MAT7 "metal(vec3(0.9, 0.6, 0.2), 0.1)"
#define MAT8 "dielectric(1.5)"
#define MAT9 "lambertian(vec3(1.0, 0.0, 0.0))"
#define MAT10 "metal(vec3(255.0, 192.0, 203.0) / 255.0, 0)"
#define MAT11 "metal(vec3(0.4, 0.5, 0.2), 0.2)"

//#define RANDOM_LAM new lambertian(vec3(CURAND(_s), CURAND(_s), CURAND(_s)))
//#define RANDOM_METAL new metal(vec3(CURAND(_s), CURAND(_s), CURAND(_s)), CURAND(_s))

__global__ void material_init(material** mat, curandState* _s) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		mat[0] = new metal(vec3(0.8, 0.85, 0.88), 0.02);;
		//mat[1] = new metal(vec3(.6, .5, .3), 0.1);
		mat[1] = new diffusive_light(vec3(20.0, 20.0, 20.0));;
		mat[2] = new dielectric(0.8);
		mat[3] = new diffusive_light(vec3(20.0, 20.0, 20.0));
		mat[4] = new metal(vec3(1.0, 182.0 / 255.0, 193.0 / 255.0), 0.01);
		mat[5] = new metal(vec3(0.8, 0.85, 0.88), 0.0);
		mat[6] = new metal(vec3(0.8, 0.6, 0.2), 0.01);
		mat[7] = new dielectric(1.5);
		mat[8] = new lambertian(vec3(0.65, 0.05, 0.05));
		mat[9] = new lambertian(vec3(0.12, 0.45, 0.15));
		mat[10] = new lambertian(vec3(0.73, 0.73, 0.73));
		mat[11] = new lambertian(vec3(102.0 / 255.0, 178.0 / 255.0, 255.0 / 255.0));
	}
}
//
//__global__ void material_update(material** mat, int idx,) {
//	if (threadIdx.x == 0 && blockIdx.x == 0) {
//		mat[0] = new lambertian(vec3(0.8, 0.8, 0.9));
//		mat[1] = new metal(vec3(5.0, 5.0, 5.0f), 0.1);
//		mat[2] = new dielectric(0.4);
//		mat[3] = new diffusive_light(vec3(1.0, 1.0, 1.0));
//		mat[4] = new lambertian(vec3(1.0, 0.75, 0.95));
//		mat[5] = new lambertian(vec3(CURAND(_s), CURAND(_s), CURAND(_s)));
//		mat[6] = new metal(vec3(0.9, 0.6, 0.2), 0.1);
//		mat[7] = new dielectric(1.5);
//		mat[8] = new lambertian(vec3(1.0, 0.0, 0.0));
//		mat[9] = new metal(vec3(255.0, 192.0, 203.0) / 255.0, 0);
//		mat[10] = new metal(vec3(0.4, 0.5, 0.2), 0.2);
//	}
//}

PbrtEngine::PbrtEngine(int _nx, int _ny, int _ns): nx(_nx), ny(_ny), ns(_ns) {
	uponPixelNumChange();
}

#define CUFREE_IF_NOT_NULL(X) if (X) {cudaFree(X); X = nullptr;}
void PbrtEngine::refreshPixelNum() {
	num_pixels = nx * ny;
	CUFREE_IF_NOT_NULL(d_rand_state)
	CUFREE_IF_NOT_NULL(d_rand_state2)
	checkCudaErrors(cudaMalloc((void**)(&d_rand_state), num_pixels * sizeof(curandState)));
	checkCudaErrors(cudaMalloc((void**)(&d_rand_state2), sizeof(curandState)));

	rand_init << <1, 1 >> > (d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	fb_size = num_pixels * sizeof(vec3);
	delete h_fb;
	h_fb = new vec3[num_pixels];
	CUFREE_IF_NOT_NULL(fb)
	checkCudaErrors(cudaMalloc((void**)(&fb), fb_size));
}


void PbrtEngine::__renderScene(bool _buildScene = true) {
	std::lock_guard<mutex> lg(mutex);
	
	m_strRenderStatus = "PbrtEgnine status: Rendering.";
	//m_vecTriangles.clear();
	if (_buildScene) {
		std::cout << "INFO: generate scene.\n";
		m_vecTriangles.clear();
		__genTrianglesFromScene();
		__genBvhNodes();
	}
	else {
		std::cout << "INFO: using scene generated.\n";
	}
	
	auto& glCam = Camera::getInstance();
	camera_update << <1, 1 >> > (cam, nx, ny, cvtGlmVec3ToVec3(glCam.m_v3Position),
		cvtGlmVec3ToVec3(glCam.m_v3Position + glCam.m_v3Front), glCam.m_fZoom, glCam.getDistToFocus(), glCam.getAperture());

	clock_t start, stop;
	start = clock();
	dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty);
	dim3 threads(tx, ty);
	//why black pic if vecBvhNodesNum == 0
	render << <blocks, threads >> > (fb, nx, ny, ns, d_bvh_nodes, vecBvhNodesNum, m_vecTriangles.size(), d_triangles, cam, d_rand_state,
		h_visitedFlags, h_stack, h_retStack, d_mats, d_texObjs, cubeMapObj);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	stop = clock();
	double seconds_taken = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "took " << seconds_taken << " seconds.\n";

	checkCudaErrors(cudaMemcpy(h_fb, fb, num_pixels * sizeof(vec3), cudaMemcpyDeviceToHost));
	std::ofstream ofs("./output.ppm");
	ofs << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = ny - 1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			size_t pixel_index = j * nx + i;
			int ir = int(255.99 * h_fb[pixel_index].r());
			int ig = int(255.99 * h_fb[pixel_index].g());
			int ib = int(255.99 * h_fb[pixel_index].b());
			ofs << ir << " " << ig << " " << ib << "\n";
		}
	}

	m_strRenderStatus = "PbrtEgnine status: finished";
	std::cout << "PbrtEngine: rendering finished.\n";
}

void PbrtEngine::renderScene(bool _buildScene) {
	/*if (worker.joinable()) {
		std::cout << "INFO: join last worker.\n";
		worker.join();
		std::cout << "INFO: finished joining worker.\n";
	}*/
	std::cout << "rendering a " << nx << "[width]x" << ny << "[height]" << " image with " << ns << " samples per pixel in ";
	std::cout << tx << "x" << ty << " blocks.\n";

	worker = std::thread(&PbrtEngine::__renderScene, this, _buildScene);
}

void PbrtEngine::join() {
	if (worker.joinable()) {
		std::cout << "INFO: join last worker.\n";
		worker.join();
		std::cout << "INFO: finished joining worker.\n";
	}
}

void PbrtEngine::addMesh(std::vector<Mesh*> _m) {
	m_vecScene = _m;
}

void PbrtEngine::addMesh(Mesh* _m) {
	m_vecScene.push_back(_m);
}

void PbrtEngine::__genTrianglesFromScene() {
	std::string target = "light";
	for (auto* p : m_vecScene) {
		if (p->getName() != target) continue;
		p->genTriangles(m_vecTriangles);
	}

	for (auto* p : m_vecScene) {
		if (p->getName() == target) continue;
		p->genTriangles(m_vecTriangles);
	}
}

void PbrtEngine::__genBvhNodes() {
	auto num = m_vecTriangles.size();
	std::cout << "Total num of Triangles: " << num << std::endl;
	if (num == 0) {
		return;
	}
	//prepare triangles end

	std::vector<PtrIdxPair> vecPtrIdxPair;
	for (int i = 0; i < num; i++) {
		vecPtrIdxPair.push_back(PtrIdxPair(&m_vecTriangles[i], i));
	}
	std::vector<bvh_node> vecBvhNodes;
	
	//make_bvh_nodes(vecPtrIdxPair.data(), num, 0.0f, 1.0f, vecBvhNodes);
	make_bvh_nodes(vecPtrIdxPair, 0, num, 0.0f, 1.0f, vecBvhNodes);
	vecBvhNodesNum = vecBvhNodes.size();

	CUFREE_IF_NOT_NULL(d_bvh_nodes)
	CUFREE_IF_NOT_NULL(d_triangles)
	checkCudaErrors(cudaMalloc((void**)&d_bvh_nodes, sizeof(bvh_node) * vecBvhNodes.size()));
	checkCudaErrors(cudaMalloc((void**)&d_triangles, sizeof(Triangle) * num));
	checkCudaErrors(cudaMemcpy(d_bvh_nodes, vecBvhNodes.data(), sizeof(vecBvhNodes) * vecBvhNodes.size(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_triangles, m_vecTriangles.data(), sizeof(Triangle) * num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&cam, sizeof(camera*)));
	auto& glCam = Camera::getInstance();
	camera_init << <1, 1 >> > (cam, nx, ny, cvtGlmVec3ToVec3(glCam.m_v3Position), 
		cvtGlmVec3ToVec3(glCam.m_v3Position + glCam.m_v3Front), glCam.m_fZoom, glCam.getDistToFocus(), glCam.getAperture());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMalloc((void**)&d_mats, sizeof(material*) * MAT_NUM));
	material_init << <1, 1 >> > (d_mats, d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	//checkCudaErrors(cudaMemset(h_retStack, 0, RET_STACK_SIZE * sizeof(hit_record) * num_pixels));
}

void PbrtEngine::__allocateCudaStackMemory() {
	std::cout << "INFO: ready to free cuda stack memory if possible.\n";
	CUFREE_IF_NOT_NULL(h_visitedFlags)
	CUFREE_IF_NOT_NULL(h_visitedFlags)
	CUFREE_IF_NOT_NULL(h_stack)
	CUFREE_IF_NOT_NULL(h_retStack)
	std::cout << "INFO: cuda stack memory freed.\n";
	checkCudaErrors(cudaMalloc((void**)&h_visitedFlags, FLAG_STACK_SIZE * sizeof(bool) * num_pixels));
	checkCudaErrors(cudaMemset(h_visitedFlags, 0, FLAG_STACK_SIZE * sizeof(bool) * num_pixels));
	checkCudaErrors(cudaMalloc((void**)&h_stack, STACK_SIZE * sizeof(int) * num_pixels));
	//checkCudaErrors(cudaMemset(h_stack, 0, STACK_SIZE * sizeof(int) * num_pixels));

	checkCudaErrors(cudaMalloc((void**)&h_retStack, RET_STACK_SIZE * sizeof(hit_record) * num_pixels));
	//checkCudaErrors(cudaMemset(h_retStack, 0, RET_STACK_SIZE * sizeof(hit_record) * num_pixels));

	std::cout << "INFO: cuda stack memory allocated.\n";

}


void PbrtEngine::saveBvhNodesToFile() {
	return;
}

void PbrtEngine::genSomeMaterials() {
	return;
}

VEC_TRI loadTriangleFromMesh(std::string _fn, TriangleMesh& _mesh) {
	
	_mesh.create_scene();
	VEC_TRI tris;
	float scale = 1.0;
	vec3 translation = vec3(0.0f, 0.0f, 0.0f);
	int idx = 0;
	for (auto& f : _mesh.faces) {
		auto v1 = _mesh.vertices[f.x() - 1] * scale + translation;
		auto v2 = _mesh.vertices[f.y() - 1] * scale + translation;
		auto v3 = _mesh.vertices[f.z() - 1] * scale + translation;
		tris.push_back(Triangle(v1, v2, v3, f.matType));
	}

	_mesh.clear();
	scale = 2.0f;
	translation = vec3(0.0f, 0.0f, 0.0f);
	loadObj(_fn, _mesh);
	int faceLimit = 2000;
	int faceCnt = 0;
	for (auto& f : _mesh.faces) {
		if (faceCnt++ > faceLimit) break;
		auto v1 = _mesh.vertices[f.x() - 1] * scale + translation;
		auto v2 = _mesh.vertices[f.y() - 1] * scale + translation;
		auto v3 = _mesh.vertices[f.z() - 1] * scale + translation;
		//tris.push_back(Triangle(v1, v2, v3, f.matType));
		tris.push_back(Triangle(v1, v2, v3, 3));
	}
	auto center = _mesh.getWeightCenter();
	auto minmax = _mesh.getMinMax();

	/*vec3 center1(0, -1000, 0);
	tris.push_back(Triangle(center1, 1000, 0));

	vec3 center2(5.8, 0.2, 1.2);
	tris.push_back(Triangle(center2, 0.2, 0));

	vec3 center3(7.5, 0.2, 0.5);
	tris.push_back(Triangle(center3, 0.2, 6));

	vec3 center4(7.8, 0.2, 1.5);
	tris.push_back(Triangle(center4, 0.25, 7));

	vec3 center5(7.8, 0.5, 2.7);
	tris.push_back(Triangle(center5, 0.5, 7));

	vec3 center6(8, 0.8, 4.5);
	tris.push_back(Triangle(center6, 0.8, 9));

	vec3 center7(8.0, 0.2, -0.5);
	tris.push_back(Triangle(center7, 0.2, 10));*/

	

	return std::move(tris);
}

void make_bvh_nodes(PtrIdxPair *_p, int n, float time0, float time1, std::vector<bvh_node> &vecBvh) {
	int axis = int(3 * drand48());
	if (axis == 0) {
		qsort(_p, n, sizeof(PtrIdxPair), box_x_compare);
	}
	else if (axis == 1) {
		qsort(_p, n, sizeof(PtrIdxPair), box_y_compare);
	}
	else {
		qsort(_p, n, sizeof(PtrIdxPair), box_z_compare);
	}

	if (n == 1) {
		vecBvh.push_back(bvh_node(_p[0]));
	}
	else if (n == 2) {
		auto idx1 = vecBvh.size();
		auto idx2 = idx1 + 1;
		vecBvh.push_back(bvh_node(_p[0])); 
		vecBvh.push_back(bvh_node(_p[1]));
		vecBvh.push_back(bvh_node(idx1, idx2, vecBvh[idx1], vecBvh[idx2]));
	}
	else {
		make_bvh_nodes(_p, n / 2, time0, time1, vecBvh);
		auto idx1 = vecBvh.size() - 1; 
		make_bvh_nodes(_p + n / 2, n - n / 2, time0, time1, vecBvh);
		auto idx2 = vecBvh.size() - 1;
		vecBvh.push_back(bvh_node(idx1, idx2, vecBvh[idx1], vecBvh[idx2]));
	}
}

void make_bvh_nodes(std::vector<PtrIdxPair> &_p, int m, int n, float time0, float time1, std::vector<bvh_node>& vecBvh) {
	int axis = int(3 * drand48());
	if (axis == 0) {
		std::sort(_p.begin() + m, _p.begin() + n, X_CMP());
	}
	else if (axis == 1) {
		std::sort(_p.begin() + m, _p.begin() + n, Y_CMP());
	}
	else {
		std::sort(_p.begin() + m, _p.begin() + n, Z_CMP());
	}

	int cnt = n - m;
	if (cnt == 1) {
		vecBvh.push_back(bvh_node(*(_p.begin()+m)));
	}
	else if (cnt == 2) {
		auto idx1 = vecBvh.size();
		auto idx2 = idx1 + 1;
		vecBvh.push_back(bvh_node(*(_p.begin() + m)));
		vecBvh.push_back(bvh_node(*(_p.begin() + m + 1)));
		vecBvh.push_back(bvh_node(idx1, idx2, vecBvh[idx1], vecBvh[idx2]));
	}
	else if (cnt != 0) {
		make_bvh_nodes(_p, m, m + cnt / 2, time0, time1, vecBvh);
		auto idx1 = vecBvh.size() - 1;
		make_bvh_nodes(_p, m + cnt / 2, n, time0, time1, vecBvh);
		auto idx2 = vecBvh.size() - 1;
		vecBvh.push_back(bvh_node(idx1, idx2, vecBvh[idx1], vecBvh[idx2]));
	}
	else {
		std::cout << "Error in make_bvh_nodes: m == n !!!.\n";
	}
}


__device__ bool hit(const ray& r, const bvh_node* world, int bvh_num, int tri_num, bool *flags, int *stack, hit_record *retStack, const Triangle *tri, float t_min, float t_max, hit_record &rec) {
#ifdef LINEAR_ITER
	for (int i = 0; i < tri_num; i++) {
		const Triangle& curr = tri[i];
		if (curr.hit(r, t_min, t_max, rec)) {
			t_max = ffmin(t_max, rec.t);
		}
	}
	return rec.isValid();
#endif
	//###########################
	int stackPtr = 0;
	int recordPtr = 0;
	int flagPtr = 0;
	int curr = bvh_num - 1;
	bool metLeaf = false;
	bool branch = true;
	while ((curr != -1) || stackPtr) {
		metLeaf = false;
		while (curr != -1) {
			auto& currNode = world[curr];
			stack[stackPtr++] = curr;
		    flags[flagPtr++] = branch;
			if (!currNode.bbx.hit(r, t_min, t_max)) {
				hit_record& top = retStack[recordPtr++];
				top.reset();
				curr = -1;
				metLeaf = true;
				continue;
			}

			if (currNode.isLeaf()) {
				tri[currNode.left].hit(r, t_min, t_max, retStack[recordPtr++]);
				metLeaf = true;
				curr = -1;
			}
			else {
				curr = currNode.left;
				branch = true;
			}
		}
		

		int top = stack[--stackPtr];
		bool currFlag = flags[--flagPtr];
		if (!metLeaf) {
			/*flags[world[top].left] = false;
			flags[world[top].right] = false;
			*/
			
			auto& left = retStack[--recordPtr];
			auto& right = retStack[--recordPtr];
			auto& rec = retStack[recordPtr++];
			
			if (left.isValid() && right.isValid()) {
				if (left.t < right.t) {
					rec = left;
				}
				else {
					rec = right;
				}
			}
			else if (right.isValid()) {
				rec = right;
			}
			else if (left.isValid()) {
				rec = left;
			}
		}

		if (stackPtr) {
			auto& top = stack[stackPtr - 1];
			/*if (!flags[world[top].right]) {
				curr = world[top].right;
			}*/

			if (currFlag) {
				curr = world[top].right;
				branch = false;
			}
		}
	}
	//flags[bvh_num - 1] = false; //bug bug bug data race, pollute other's stack
	rec = *retStack;
	return retStack->isValid();
}

#define ITER_NUM 50
__device__ vec3 color(const ray& r, const bvh_node* world, int bvh_num, int tri_num, const Triangle *tri, curandState* s, 
	bool* visitedFlags, int* stack, hit_record* retStack, material** mats, TEX_OBJ _texObj, cudaTextureObject_t _cubeMap) {
	
	//lambertian mat(vec3(0.0f, 1.0f, 0.0f)); // here material is fixed
	ray cur_ray = r;
	const Triangle* light = tri; //hard code light as the frist !!!
	vec3 retColor = vec3(1.0, 1.0, 1.0);
	float pdf;
	for (int i = 0; i < ITER_NUM; i++) {
		hit_record rec;
		vec3 emitted;
		if (hit(cur_ray, world, bvh_num, tri_num, visitedFlags, stack, retStack, tri, 0.001f, FLT_MAX, rec)) {
			/*if (!rec.hitFront) {
				return vec3(1.0, 0.0, 0.0);
			}*/
			
			scatter_record srec; 
			srec.pdf_ptr = nullptr;
			vec3 attenuation; //not useful
			ray scattered;
			material *mat = mats[rec.matType];
			emitted = mat->emitted(0, 0);
			
			if (mat->scatter(cur_ray, rec, attenuation, srec, s, _texObj)) {
				
				if (srec.is_specular) {
					retColor *= srec.attenuation;
					scattered = srec.specular_ray;
				}
				else {
#if defined(OLD)
					//old scatter
					vec3 target = rec.p + rec.normal + random_in_unit_sphere(s);
					scattered = ray(rec.p, target - rec.p, cur_ray.time());
					retColor *= srec.attenuation;
#else
					/*hittable_pdf p2 = hittable_pdf(light, rec.p);
					mixture_pdf p = mixture_pdf(srec.pdf_ptr, &p2);
					scattered = ray(rec.p, p.generate(s), r.time());
					pdf = p.value(scattered.direction());*/
					//return vec3(0, 0, 0);
					cosine_pdf p1 = *(cosine_pdf*)srec.pdf_ptr;
					
					hittable_pdf p2 = hittable_pdf(light, rec.p);
					
					float thres = 0.5;
					if (CURAND(s) < thres) {
						scattered = ray(rec.p, p1.generate(s), r.time());
					}
					else {
						scattered = ray(rec.p, p2.generate(s), r.time());
					}
					
					//scattered = ray(rec.p, p.generate(s), r.time());
					
					/*if (dot(scattered.direction(), rec.normal) < 0.0) {
						return vec3(0.0, 0.0, 0.0);
					}*/
					//pdf = p.value(scattered.direction());
					pdf = thres * p1.value(scattered.direction()) + (1 - thres) * p2.value(scattered.direction());
					if (pdf < 1e-6) {
						return vec3(0.0, 0.0, 0.0);
					}
					retColor *= srec.attenuation * mat->scatter_pdf(cur_ray, rec, scattered) / pdf;
					if (srec.pdf_ptr)
						delete srec.pdf_ptr;
#endif
				}
				cur_ray = scattered;
			}
			else {
				/*if (rec.matType == 0) {
					return vec3(1.0, 0.0, 0.0);
				}
				else if (rec.matType == 1) {
					return vec3(0.0, 1.0, 0.0);
				}
				else if (rec.matType == 2) {
					return vec3(0.0, 0.0, 1.0);
				}
				else {
					return vec3(1.0, 1.0, 1.0);
				}*/
				//retColor *= emitted;
				retColor *= srec.attenuation;
				goto RETURN_COLOR;
			}
		}
		else {
			/*vec3 unit_direction = unit_vector(cur_ray.direction());
			float4 envMap = texCubemap<float4>(_cubeMap, unit_direction.x(), unit_direction.y(), unit_direction.z());
			vec3 c = vec3(envMap.x, envMap.y, envMap.z);
			return cur_attenuation * c;*/
			//old below
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f * (unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
			retColor *= c;
			goto RETURN_COLOR;
		}
	}

RETURN_COLOR:

	clipColor(retColor);
	return retColor;
}


__global__ void render(vec3* fb, int max_x, int max_y, int ns, const bvh_node* bvh, int bvh_num, int tri_num, Triangle* tri, camera **cam, curandState* randState,
	bool *h_visitedFlags, int *h_stack, hit_record *h_retStack, material **mat, TEX_OBJ _texObj, cudaTextureObject_t _cubeMap) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	bool* visitedFlags = pixel_index * FLAG_STACK_SIZE + h_visitedFlags;
	int* stack = pixel_index * STACK_SIZE + h_stack;
	hit_record* retStack = pixel_index * RET_STACK_SIZE + h_retStack;

	curand_init(1995 + pixel_index, 0, 0, &randState[pixel_index]);
	curandState local_state = randState[pixel_index];
	vec3 col; 
	for (int k = 0; k < ns; k++){
		float u = float(i + curand_uniform(&local_state)) / float(max_x);
		float v = float(j + curand_uniform(&local_state)) / float(max_y);
		ray r = (*cam)->get_ray(u, v, &local_state);
		col += color(r, bvh, bvh_num, tri_num, tri, &local_state, visitedFlags, stack, retStack, mat, _texObj, _cubeMap);
	}

	col /= float(ns);
	for (int i = 0; i < 3; i++) {
		col[i] = col[i] > 1.0 ? 1.0 : col[i];
	}
	fb[pixel_index] = col;
}

__global__ void rand_init(curandState* rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1995, 0, 0, rand_state);
	}
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
	int curr_x = blockIdx.x * blockDim.x + threadIdx.x;
	int curr_y = blockIdx.y * blockDim.y + threadIdx.y;
	if (curr_x >= max_x || curr_y >= max_y) return;
	int pixel_idx = curr_y * max_x + curr_x;
	curand_init(1995 + pixel_idx, 0, 0, &rand_state[pixel_idx]);
}

__global__ void camera_init(camera** cam, int nx, int ny) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		vec3 lookfrom(13, 2, 3);
		lookfrom = vec3(5.5, 2.5, 8);
		//vec3 lookat(2, 1, 0);
		//vec3 lookat(0, 0, 0);
		vec3 lookat(0, 1.3, 0); //lastest
		//float dist_to_focus = (lookfrom - lookat).length();
		float dist_to_focus = 10.0;
		float aperture = 0.0;
		//camera cam(lookfrom, lookat, vec3(0, 1, 0), 20, float(nx) / float(ny), aperture, dist_to_focus, 0.0, 1.0);
		*cam = new camera(lookfrom, lookat, vec3(0, 1, 0), 40, float(nx) / float(ny), aperture, dist_to_focus, 0.0, 1.0);
	}
}

__global__ void camera_update(camera** cam, int nx, int ny, vec3 lookfrom, vec3 lookat, float vfov, 
	float dist_to_focus, float aperture) {
	if(threadIdx.x == 0 && blockIdx.x == 0) {
		(*cam)->update(lookfrom, lookat, vec3(0, 1, 0), 40, float(nx) / float(ny), aperture, dist_to_focus, 0.0, 1.0);
		//camera cam(lookfrom, lookat, vec3(0, 1, 0), 20, float(nx) / float(ny), aperture, dist_to_focus, 0.0, 1.0);
	}
}

__global__ void camera_init(camera** cam, int nx, int ny, vec3 lookfrom, vec3 lookat, float vfov,
	float dist_to_focus, float aperture) {
	
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		//float dist_to_focus = (lookfrom - lookat).length();
		//camera cam(lookfrom, lookat, vec3(0, 1, 0), 20, float(nx) / float(ny), aperture, dist_to_focus, 0.0, 1.0);
		*cam = new camera(lookfrom, lookat, vec3(0, 1, 0), vfov, float(nx) / float(ny), aperture, dist_to_focus, 0.0, 1.0);
	}
}

void pushMoreTriangles(std::vector<Triangle> &_triVec) {
	vec3 center1(0, -1000, 0);
	_triVec.push_back(Triangle(center1, 1000, 0));

	float xShift = -4.0;

	vec3 center2(5.8 + xShift, 0.5, 1.2);
	_triVec.push_back(Triangle(center2, 0.5, 7));

	vec3 center3(7.5 + xShift, 0.6, 0.5);
	_triVec.push_back(Triangle(center3, 0.6, 7));

	vec3 center4(7.8 + xShift, 0.2, 1.5);
	_triVec.push_back(Triangle(center4, 0.25, 7));

	vec3 center5(7.8 + xShift, 0.5, 2.7);
	_triVec.push_back(Triangle(center5, 0.5, 7));

	vec3 center6(8.0 + xShift, 0.8, 4.5);
	_triVec.push_back(Triangle(center6, 0.8, 7));

	vec3 center7(8.0 + xShift, 0.2, -0.5);
	_triVec.push_back(Triangle(center7, 0.2, 7));

	float lw = 1.2;
	float lh = 0.9;
	lw = 10.0f;
	lh = 10.0f;
	///*float lw = 5.0;
	//float lh = 5.0;*/
	//material* lptr = new diffuse_light(new constant_texture(vec3(5, 5, 5)));
	//list[i++] = new xz_rect(8 - lw, 8 + lw, 1 - lh, 1 + lh, 3, lptr);

	float posX = 0.0;
	float posY = 3.0;
	float posZ = 1.0;
	auto p0 = vec3(posX - lw, posY + 0.1, posZ - lh);
	auto p1 = vec3(posX - lw, posY, posZ + lh);
	auto p2 = vec3(posX + lw, posY + 0.1, posZ + lh);
	auto p3 = vec3(posX + lw, posY, posZ - lh);

	_triVec.push_back(Triangle(p0, p1, p2, 3));
	_triVec.push_back(Triangle(p0, p2, p3, 3));

}

//int rt_main() {
//	int nx = 600;
//	int ny = 300;
//	int ns = 1;
//
//	/*nx = 1920 / 2;
//	ny = 1080 / 2;
//	ns = 50;*/
//	int tx = 16;
//	int ty = 16;
//
//	std::cout << "rendering a " << nx << "[width]x" << ny << "[height]" << " image with " << ns << " samples per pixel in ";
//	std::cout << tx << "x" << ty << " blocks.\n";
//
//	int num_pixels = nx * ny;
//	size_t fb_size = num_pixels * sizeof(vec3);
//
//	vec3* fb;
//	vec3* h_fb = new vec3[num_pixels];
//	checkCudaErrors(cudaMalloc((void**)(&fb), fb_size));
//
//	curandState* d_rand_state;
//	checkCudaErrors(cudaMalloc((void**)(&d_rand_state), num_pixels * sizeof(curandState)));
//	curandState* d_rand_state2;
//	checkCudaErrors(cudaMalloc((void**)(&d_rand_state2), sizeof(curandState)));
//
//	rand_init<<<1, 1>>>(d_rand_state2);
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	//prepare triangles
//	/*TriangleMesh mesh;
//	VEC_TRI triangles = std::move(loadTriangleFromMesh("./models/monkey.obj", mesh));*/
//
//	auto m = RtModel("./models/bunny.obj", 7, 16.0, vec3(1.0, -0.5, 3.0));
//	auto triangles = m.genTriangleVector();
//	pushMoreTriangles(triangles);
//
//	auto num = triangles.size();
//	std::cout << "Total num of Triangles: " << num << std::endl;
//	//prepare triangles end
//
//	std::vector<PtrIdxPair> vecPtrIdxPair;
//	for (int i = 0; i < num; i++) {
//		vecPtrIdxPair.push_back(PtrIdxPair(&triangles[i], i));
//	}
//	std::vector<bvh_node> vecBvhNodes;
//	//make_bvh_nodes(vecPtrIdxPair.data(), num, 0.0f, 1.0f, vecBvhNodes);
//	make_bvh_nodes(vecPtrIdxPair, 0, num, 0.0f, 1.0f, vecBvhNodes);
//	auto& bvhTop = vecBvhNodes.back();
//
//	bvh_node* d_bvh_nodes;
//	Triangle* d_triangles;
//	checkCudaErrors(cudaMalloc((void**)&d_bvh_nodes, sizeof(bvh_node) * vecBvhNodes.size()));
//	checkCudaErrors(cudaMalloc((void**)&d_triangles, sizeof(Triangle) * num));
//	checkCudaErrors(cudaMemcpy(d_bvh_nodes, vecBvhNodes.data(), sizeof(vecBvhNodes) * vecBvhNodes.size(), cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpy(d_triangles, triangles.data(), sizeof(Triangle) * num, cudaMemcpyHostToDevice));
//
//	camera** cam;
//	checkCudaErrors(cudaMalloc((void**)&cam, sizeof(camera*)));
//	camera_init<<<1, 1 >>>(cam, nx, ny);
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());
//
//
//	material** d_mats;
//	checkCudaErrors(cudaMalloc((void**)&d_mats, sizeof(material*) * MAT_NUM));
//	material_init << <1, 1 >> > (d_mats, d_rand_state2);
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());
// 
//	bool* h_visitedFlags;
//	checkCudaErrors(cudaMalloc((void**)&h_visitedFlags, FLAG_STACK_SIZE * sizeof(bool) * num_pixels));
//	checkCudaErrors(cudaMemset(h_visitedFlags, 0, FLAG_STACK_SIZE * sizeof(bool) * num_pixels));
//	int* h_stack;
//	checkCudaErrors(cudaMalloc((void**)&h_stack, STACK_SIZE * sizeof(int) * num_pixels));
//	//checkCudaErrors(cudaMemset(h_stack, 0, STACK_SIZE * sizeof(int) * num_pixels));
//	hit_record* h_retStack; 
//	checkCudaErrors(cudaMalloc((void**)&h_retStack, RET_STACK_SIZE * sizeof(hit_record) * num_pixels));
//	//checkCudaErrors(cudaMemset(h_retStack, 0, RET_STACK_SIZE * sizeof(hit_record) * num_pixels));
//
//	clock_t start, stop;
//	start = clock();
//	dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty);
//	dim3 threads(tx, ty);
//	render<<<blocks, threads>>> (fb, nx, ny, ns, d_bvh_nodes, vecBvhNodes.size(), num, d_triangles, cam, d_rand_state,
//				h_visitedFlags, h_stack, h_retStack, d_mats);
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	stop = clock();
//	double seconds_taken = ((double)(stop - start)) / CLOCKS_PER_SEC;
//	std::cout << "took " << seconds_taken << " seconds.\n";
//
//	checkCudaErrors(cudaMemcpy(h_fb, fb, num_pixels * sizeof(vec3), cudaMemcpyDeviceToHost));
//	std::ofstream ofs("./output.ppm");
//	ofs << "P3\n" << nx << " " << ny << "\n255\n";
//	for (int j = ny - 1; j >= 0; j--) {
//		for (int i = 0; i < nx; i++) {
//			size_t pixel_index = j * nx + i;
//			int ir = int(255.99 * h_fb[pixel_index].r());
//			int ig = int(255.99 * h_fb[pixel_index].g());
//			int ib = int(255.99 * h_fb[pixel_index].b());
//			ofs << ir << " " << ig << " " << ib << "\n";
//		}
//	}
//
//	return 0;
//}