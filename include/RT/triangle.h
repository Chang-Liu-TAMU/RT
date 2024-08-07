#pragma once
#include <vector>
#include <fstream>
#include <string>
#include <exception>
#include <utility>
#include <cfloat>
#include "material.h"
#include "climits"
#include "macros.h"

inline float drand48() {
	float ret = rand() / (float)RAND_MAX;
	return ret;
}

inline __host__ __device__ float ffmin(float a, float b) {
	return a < b ? a : b;
}

inline __host__ __device__ float ffmax(float a, float b) {
	return a > b ? a : b;
}

class aabb {
public:
	__host__ __device__ aabb() {}
	__host__ __device__ aabb(const vec3& a, const vec3& b) {
		_min = a + vec3(-SHIFT, -SHIFT, -SHIFT);
		_max = b + vec3(SHIFT, SHIFT, SHIFT);
	}

	__host__ __device__ vec3 min() const { return _min; }
	__host__ __device__ vec3 max() const { return _max; }

	/*bool hit(const ray& r, float tmin, float tmax) const {
		for (int a = 0; a < 3; a++) {
			float x = (_min[a] - r.origin()[a]) / r.direction()[a];
			float y = (_max[a] - r.origin()[a]) / r.direction()[a];
			float t0 = ffmin(x, y);
			float t1 = ffmax(x, y);
			tmin = ffmax(t0, tmin);
			tmax = ffmin(t1, tmax);
			if (tmax <= tmin) {
				return false;
			}
			return true;
		}

	}*/

	__host__ __device__ bool hit(const ray& r, float tmin, float tmax) const {
		for (int a = 0; a < 3; a++) {
			float invD = 1.0f / r.direction()[a];
			float t0 = (_min[a] - r.origin()[a]) * invD;
			float t1 = (_max[a] - r.origin()[a]) * invD;
			if (invD < 0.0f) {
				float tem = t0;
				t0 = t1;
				t1 = tem;
			}
			tmin = t0 > tmin ? t0 : tmin;
			tmax = t0 < tmax ? t1 : tmax;
			if (tmax <= tmin) {
				return false;
			}
			// return true; bug
		}
		return true;
	}

	vec3 _min;
	vec3 _max;
};

inline __host__ __device__ aabb surrounding_box(const aabb& box0, const aabb& box1) noexcept {
	vec3 small(fmin(box0.min().x(), box1.min().x()),
		fmin(box0.min().y(), box1.min().y()),
		fmin(box0.min().z(), box1.min().z()));
	vec3 big(fmax(box0.max().x(), box1.max().x()),
		fmax(box0.max().y(), box1.max().y()),
		fmax(box0.max().z(), box1.max().z()));

	return aabb(small, big);
}

class Triangle {
public:
	__host__ __device__ Triangle() {}
	__host__ __device__ Triangle(vec3 _v0, vec3 _v1, vec3 _v2, int matType, int _texIdx = 0,
						vec3 _u = vec3(), vec3 _v = vec3() ): texIdx(_texIdx), u(_u), v(_v) {
		setTriangle(_v0, _v1, _v2, matType);
	}

	__host__ __device__ Triangle(vec3 _o, float _r, int _matType) {
		matType = _matType;
		v0 = _o;
		v1[0] = _r;
		shapeType = 1;
	}

	__host__ __device__ void setTriangle(vec3 _v0, vec3 _v1, vec3 _v2, int matType = 0) {
		if (shapeType != 0) return;
		v0 = _v0;
		v1 = _v1;
		v2 = _v2;
		edge1 = v1 - v0;
		edge2 = v2 - v0; 
		normal = unit_vector(cross(edge1, edge2));
		this->matType = matType;
	}

	__host__ __device__ void setUV(float x, float y, hit_record& rec) const noexcept {
		float z = 1.0 - x - y;
		rec.u = z * u[0] + x * u[1] + y * u[2];
		rec.v = z * v[0] + x * v[1] + y * v[2];
		return;
	}

	__host__ __device__ void setNormal(vec3 _n1, vec3 _n2, vec3 _n3) noexcept {
		n0 = _n1; 
		n1 = _n2;
		n2 = _n3;
		calNormal = 0;
	}

	__host__ __device__ vec3 resloveNorm(float x, float y) const noexcept {
		float z = 1.0 - x - y;
		return z * n0 + x * n1 + y * n2;
	}

	__host__ __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const noexcept {
#ifndef LINEAR_ITER
		rec.reset();
#endif
		if (shapeType == 0) {
			vec3 tvec = r.origin() - v0;
			vec3 pvec = cross(r.direction(), edge2);
			float det = dot(edge1, pvec);
			det = 1.0f / det;
			float epsilon = 1e-6;
			float u = dot(tvec, pvec) * det;
			if (u < 0.0f || u > 1.0f) {
				return false;
			}
			vec3 qvec = cross(tvec, edge1);
			float v = dot(r.direction(), qvec) * det;

			if (v < 0.0f || (u + v) > 1.0f) {
				return false;
			}

			float t = dot(edge2, qvec) * det;
			if (t <= t_min || t >= t_max) {
				return false;
			}

			vec3 currNorm = normal;
			if (calNormal == 0) {
				vec3 currNorm = resloveNorm(u, v);
			}
			
			rec.t = t;

			rec.p = r.point_at_parameter(t);
			if (dot(r.direction(), currNorm) < 0.0f) {
				rec.normal = currNorm;
				rec.hitFront = true;
			}
			else {
				rec.normal = -currNorm;
				rec.hitFront = false;
			}
			rec.matType = matType;
			rec.state = 0;
			setUV(u, v, rec);
			rec.texIdx = texIdx;
			return true;
		}
		else if (shapeType == 1) {
			auto center = v0;
			float radius = v1[0];
			vec3 oc = r.origin() - center;
			float a = dot(r.direction(), r.direction());
			float b = 2.0 * dot(oc, r.direction());
			float c = dot(oc, oc) - radius * radius;
			float discriminant = b * b - 4 * a * c;
			if (discriminant > 0) {
				float tem = (-b - sqrt(discriminant)) / (2.0 * a);
				if (tem < t_max && tem > t_min) {
					rec.t = tem;
					rec.p = r.point_at_parameter(rec.t);
					rec.normal = (rec.p - center) / radius;
					rec.matType = matType;
					rec.state = 0;
					rec.hitFront = true;
					rec.texIdx = -1;
					return true;
				}
				tem = (-b - sqrt(discriminant)) / (2.0 * a);
				if (tem < t_max && tem > t_min) {
					rec.t = tem;
					rec.p = r.point_at_parameter(rec.t);
					rec.normal = (rec.p - center) / radius;
					rec.matType = matType;
					rec.state = 0;
					rec.hitFront = true;
					rec.texIdx = -1;
					return true;
				}
			}
			return false;
		}
		else {
			return false;
		}
		
	};

	__host__ __device__ bool bounding_box(float t0, float t1, aabb& box) const {
		if (shapeType == 0)
		{
			float x1 = ffmin(ffmin(v0.x(), v1.x()), v2.x());
			float x2 = ffmax(ffmax(v0.x(), v1.x()), v2.x());

			float y1 = ffmin(ffmin(v0.y(), v1.y()), v2.y());
			float y2 = ffmax(ffmax(v0.y(), v1.y()), v2.y());

			float z1 = ffmin(ffmin(v0.z(), v1.z()), v2.z());
			float z2 = ffmax(ffmax(v0.z(), v1.z()), v2.z());

			auto p1 = vec3(x1, y1, z1);
			auto p2 = vec3(x2, y2, z2);
			box = aabb(p1, p2);
			return true;
		}
		else if (shapeType == 1)
		{
			auto &center = v0;
			auto radius = v1[0];
			box = aabb(center - vec3(radius, radius, radius), center + vec3(radius, radius, radius));
			return true;
		}
		else {
			return true;
		}
		
	};

	__host__ __device__ aabb bounding_box() const {
		if (shapeType == 0) {
			float x1 = ffmin(ffmin(v0.x(), v1.x()), v2.x());
			float x2 = ffmax(ffmax(v0.x(), v1.x()), v2.x());

			float y1 = ffmin(ffmin(v0.y(), v1.y()), v2.y());
			float y2 = ffmax(ffmax(v0.y(), v1.y()), v2.y());

			float z1 = ffmin(ffmin(v0.z(), v1.z()), v2.z());
			float z2 = ffmax(ffmax(v0.z(), v1.z()), v2.z());

			auto p1 = vec3(x1, y1, z1);
			auto p2 = vec3(x2, y2, z2);
			return aabb(p1, p2);
		}
		else if (shapeType == 1) {
			auto& center = v0;
			auto radius = v1[0];
			return aabb(center - vec3(radius, radius, radius), center + vec3(radius, radius, radius));
		}
		else {
			return aabb();
		}
	};

	__device__ float pdf_value(const vec3& o, const vec3& v) const {
		if (shapeType == 0) {
			float x0 = 213;
			float z0 = 227;
			float x1 = 343;
			float z1 = 332;
			/*float x0 = v1.x();
			float x1 = v0.x();
			float z0 = v0.z();
			float z1 = v2.z();*/
			hit_record rec;
			//if (hit(ray(o, v), 0.0001, FLT_MAX, rec)) {
			float area = (x1 - x0) * (z1 - z0);
			float distance2 = v.squared_length();
			//float cosine = cuAbs(dot(v, rec.normal) / v.length());
			float cosine = cuAbs(unit_vector(v).y());
			if (cosine < 0.000001) {
				return 0;
			}
			/*if (dot(v, rec.normal) < 0) {
				return 0;
			}*/
			return distance2 / (cosine * area);
			//}
			/*else {
				return 0;
			}*/
		}
		else if (shapeType == 1) {
			//sample to a sphere 
			hit_record rec;
			if (hit(ray(o, v), 0.001, FLT_MAX, rec)) {
				float r = v1[0];
				float cos_theta_max = sqrt(1 - r * r / (v0 - o).squared_length());
				float solid_angle = 2 * M_PI * (1 - cos_theta_max);
				return 1 / solid_angle;
			}
			else {
				return 0;
			}
		}

		return 0.0;
	}

	__device__ vec3 random(const vec3 &o,  curandState* _s) const {
		int diff = 0;
		if (shapeType == 0) {
			float x0 = 213 - diff;
			float z0 = 227 - diff;
			float x1 = 343 + diff;
			float z1 = 332 + diff;
			/*float x0 = v1.x();
			float x1 = v0.x();
			float z0 = v0.z();
			float z1 = v2.z();*/
			vec3 random_point = vec3(x0 + CURAND(_s) * (x1 - x0), 550, z0 + CURAND(_s) * (z1 - z0));
			return random_point - o;
		}
		else if (shapeType == 1) {
			//sample to a sphere 
			vec3 d = v0 - o;
			float d2 = d.squared_length();
			onb uvw;
			uvw.build_from_w(d);
			return uvw.local(random_to_sphere(v1[0], d2, _s));
		}

	}

	vec3 v0, v1, v2;
	vec3 normal;
	vec3 edge1, edge2;
	vec3 u{0.0, 0.0, 0.0};
	vec3 v{0.0, 0.0, 0.0};
	vec3 n0, n1, n2;
	int calNormal = -1;
	int matType = 0;
	int shapeType = 0; // 0 => tri; 1 => sphere
	int texIdx = 0;

};

struct PtrIdxPair {
	PtrIdxPair(Triangle* _p, int _idx) : p(_p), idx(_idx) {}
	Triangle* p;
	int idx;
};

class bvh_node {
public:
	__host__ __device__ bvh_node() {}
	__host__ __device__ bvh_node(const bvh_node &a, const bvh_node &b) {
		bbx = surrounding_box(a.bbx, b.bbx);
	}

	__host__ __device__ bvh_node(int l, int r, const bvh_node& a, const bvh_node& b) {
		left = l,
		right = r;
		bbx = surrounding_box(a.bbx, b.bbx);
	}

	__host__ __device__ bvh_node(PtrIdxPair &_p) {
		left = _p.idx;
		right = _p.idx;
		bbx = _p.p->bounding_box();
	}

	__host__ __device__ bool isLeaf() const noexcept{
		return left == right;
	}

	aabb bbx;
	int left = INT_MAX;
	int right = INT_MAX;

};


struct X_CMP {
	bool operator()(PtrIdxPair a, PtrIdxPair b) {
		aabb box_left, box_right;
		if (!a.p->bounding_box(0, 0, box_left) ||
			!b.p->bounding_box(0, 0, box_right)) {
			throw "no bounding box in bvh constructor";
		}
		if (box_left.min().x() < box_right.min().x())
			return true;
		else
			return false;
	}
};

struct Y_CMP {
	bool operator()(PtrIdxPair a, PtrIdxPair b) {
		aabb box_left, box_right;
		if (!a.p->bounding_box(0, 0, box_left) ||
			!b.p->bounding_box(0, 0, box_right)) {
			throw "no bounding box in bvh constructor";
		}
		if (box_left.min().y() < box_right.min().y())
			return true;
		else
			return false;
	}
};

struct Z_CMP {
	bool operator()(PtrIdxPair a, PtrIdxPair b) {
		aabb box_left, box_right;
		if (!a.p->bounding_box(0, 0, box_left) ||
			!b.p->bounding_box(0, 0, box_right)) {
			throw "no bounding box in bvh constructor";
		}
		if (box_left.min().z() < box_right.min().z())
			return true;
		else
			return false;
	}
};

inline int box_x_compare(const void* a, const void* b) {
	aabb box_left, box_right;
	PtrIdxPair *ah = (PtrIdxPair*)a;
	PtrIdxPair *bh = (PtrIdxPair*)b;
	if (!ah->p->bounding_box(0, 0, box_left) ||
		!bh->p->bounding_box(0, 0, box_right)) {
		throw "no bounding box in bvh constructor";
	}
	if (box_left.min().x() < box_right.min().x())
		return -1;
	else
		return 1;

}

inline int box_y_compare(const void* a, const void* b) {
	aabb box_left, box_right;
	PtrIdxPair* ah = (PtrIdxPair*)a;
	PtrIdxPair* bh = (PtrIdxPair*)b;
	if (!ah->p->bounding_box(0, 0, box_left) ||
		!bh->p->bounding_box(0, 0, box_right)) {
		throw "no bounding box in bvh constructor";
	}
	if (box_left.min().y() < box_right.min().y())
		return -1;
	else
		return 1;

}

inline int box_z_compare(const void* a, const void* b) {
	aabb box_left, box_right;
	PtrIdxPair* ah = (PtrIdxPair*)a;
	PtrIdxPair* bh = (PtrIdxPair*)b;
	if (!ah->p->bounding_box(0, 0, box_left) ||
		!bh->p->bounding_box(0, 0, box_right)) {
		throw "no bounding box in bvh constructor";
	}
	if (box_left.min().z() < box_right.min().z())
		return -1;
	else
		return 1;

}


struct TriangleFace {
	TriangleFace() {}
	TriangleFace(int x, int y, int z, int mat = 0, int base = 0) {
		matType = mat;
		v[0] = x + base;
		v[1] = y + base;
		v[2] = z + base; 
	}

	int matType = 0;
	int v[3];
	int x() {
		return v[0];
	}

	int y() {
		return v[1];
	}

	int z() {
		return v[2];
	}

	void shift(int b) {
		v[0] += b;
		v[1] += b;
		v[2] += b;
	}
};

struct TriangleMesh {

	/*std::vector<vec3> vertices = { vec3(5.0f, 2.0f, -0.8f), vec3(4.0f, 0.0f, 0.0f),
									vec3(6.0f, 0.0f, 0.0f), vec3(5.0f, 0.0f, -1.6f) };

	std::vector<TriangleFace> faces{ TriangleFace(1, 2, 3, 10), TriangleFace(1, 3, 4, 10),
									TriangleFace(1, 4, 2, 10), TriangleFace(2, 4, 3, 10) };*/

	std::vector<vec3> vertices;

	std::vector<TriangleFace> faces;

	void create_obj(float x, float y, float z, int mat = -1) {
		int base = vertices.size();
		vertices.push_back(vec3(5.0f+x, 2.0f+y, -0.8f+z));
		vertices.push_back(vec3(4.0f+x, 0.0f+y,  0.0f+z));
		vertices.push_back(vec3(6.0f+x, 0.0f+y,  0.0f+z));
		vertices.push_back(vec3(5.0f+x, 0.0f+y, -1.6f+z));
		faces.push_back(TriangleFace(1, 2, 3, drand48() * 10, base));
		faces.push_back(TriangleFace(1, 3, 4, drand48() * 10, base));
		faces.push_back(TriangleFace(1, 4, 2, drand48() * 10, base));
		faces.push_back(TriangleFace(2, 4, 3, drand48() * 10, base));
	}

	void clear() {
		vertices.clear();
		faces.clear();
	}

	void create_scene() {
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
		int base = vertices.size();
		vertices.push_back(vec3(posX - lw, posY+0.1,  posZ - lh));
		vertices.push_back(vec3(posX - lw, posY, posZ + lh));
		vertices.push_back(vec3(posX + lw, posY+0.1, posZ + lh));
		vertices.push_back(vec3(posX + lw, posY, posZ - lh));
		faces.push_back(TriangleFace(0, 1, 2, 3, base+1));
		faces.push_back(TriangleFace(0, 2, 3, 3, base+1));
		
		/*lw = 10;
		lh = 10;
		base = vertices.size();
		vertices.push_back(vec3(8 - lw, 0.1, 1 - lh));
		vertices.push_back(vec3(8 - lw, 0, 1 + lh));
		vertices.push_back(vec3(8 + lw, 0.1, 1 + lh));
		vertices.push_back(vec3(8 + lw, 0, 1 - lh));
		faces.push_back(TriangleFace(0, 1, 2, 8, base+1));
		faces.push_back(TriangleFace(0, 2, 3, 8, base+1));*/

		/*create_obj(0, 0, 2, 6);
		create_obj(2.5, 0, 0, 7);*/
		/*create_obj(0, 0, 2, 6);
		create_obj(2.5, 0, 0, 7);*/
	}

	vec3 getWeightCenter() {
		vec3 acc(0.0f, 0.0f, 0.0f);
		for (auto& v : vertices) {
			acc += v;
		}
		acc /= vertices.size();
		return acc;
	}

	std::pair<vec3, vec3> getMinMax() {
		vec3 min(FLT_MAX, FLT_MAX, FLT_MAX);
		vec3 max(FLT_MIN, FLT_MIN, FLT_MIN);
		for (auto& v : vertices) {
			min[0] = ffmin(min[0], v[0]);
			min[1] = ffmin(min[1], v[1]);
			min[2] = ffmin(min[2], v[2]);
			
			max[0] = ffmax(max[0], v[0]);
			max[1] = ffmax(max[1], v[1]);
			max[2] = ffmax(max[2], v[2]);
		}
		return { min, max };
	}

};

inline void loadObj(const std::string filename, TriangleMesh& mesh)
{
	std::ifstream in(filename.c_str());

	if (!in.good())
	{
		std::cout << "ERROR: loading obj:(" << filename << ") file not found or not good" << "\n";
		system("PAUSE");
		exit(0);
	}

	char buffer[256], str[255];
	float f1, f2, f3;

	while (!in.getline(buffer, 255).eof())
	{
		buffer[255] = '\0';
		sscanf_s(buffer, "%s", str, 255);

		// reading a vertex
		if (buffer[0] == 'v' && (buffer[1] == ' ' || buffer[1] == 32)) {
			if (sscanf_s(buffer, "v %f %f %f", &f1, &f2, &f3) == 3) {
				mesh.vertices.push_back({ f1, f2, f3 });
			}
			else {
				std::cout << "ERROR: vertex not in wanted format in OBJLoader" << "\n";
				exit(-1);
			}
		}

		// reading faceMtls 
		else if (buffer[0] == 'f' && (buffer[1] == ' ' || buffer[1] == 32))
		{
			int data1[3], data2[3], data3[3], data4[3];
			int nt = sscanf_s(buffer, "f %d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d", &data1[0], &data1[1], &data1[2],
				&data2[0], &data2[1], &data2[2], &data3[0], &data3[1], &data3[2], &data4[0], &data4[1], &data4[2]);
			
			/*int nt = sscanf_s(buffer, "f %d//%d %d//%d %d//%d", &data1[0], &data2[0], &data1[1],
				&data2[1], &data1[2], &data2[2]);

			TriangleFace f;
			f.v[0] = data1[0];
			f.v[1] = data1[1];
			f.v[2] = data1[2];
			f.matType = 10;
			mesh.faces.push_back(f);
			continue;*/

			if (nt == 12) {
				TriangleFace f;
				f.v[0] = data1[0];
				f.v[1] = data2[0];
				f.v[2] = data3[0];
				mesh.faces.push_back(f);
				f.v[0] = data1[0];
				f.v[1] = data3[0];
				f.v[2] = data4[0];
				mesh.faces.push_back(f);
			}
			else if (nt == 9) {
				TriangleFace f;
				f.v[0] = data1[0];
				f.v[1] = data2[0];
				f.v[2] = data3[0];
				mesh.faces.push_back(f);
			}
			else {
				std::cout << "ERROR: I don't know the format of that FaceMtl" << "\n";
				exit(-1);
			}
		}
	}
}