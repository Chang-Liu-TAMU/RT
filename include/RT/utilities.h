#pragma once
//#include <memory>
//#include <iostream>
//#include <algorithm>
//#include <mutex>

//#include "stdlib.h"
#include "vec3.h"
#include "texture.h"
//#include "ray.h"
//#include "material.h"
//#include "triangle.h"
//#include "rt_camera.h"
//#include "rt_model.h"
//#include "camera.h"

__device__ void getAttenuationFromTexture(cudaTextureObject_t* _t, float _u, float _v, int _texIdx, vec3& _attenu);

__device__ void getAttenuationFromTexture(CudaTexture* _t, float _u, float _v, int _texIdx, vec3& _attenu);

__device__ vec3 random_cosine_direction(curandState* _s);

__device__ vec3 random_to_sphere(float r, float d2, curandState* _s);

class onb {
public:
	vec3 axis[3];
public:
	
	D_H onb() {}

	D_H vec3 operator[](int i) const { return axis[i]; }

	D_H vec3 u() const { return axis[0]; }
	D_H vec3 v() const { return axis[1]; }
	D_H vec3 w() const { return axis[2]; }

	D_H vec3 local(float a, float b, float c) {
		return a * axis[0] + b * axis[1] + c * axis[2];
	}

	D_H vec3 local(const vec3& a) const {
		return a.x() * u() + a.y() * v() + a.z() * w();
	}

	D_H void build_from_w(const vec3& n) {
		axis[2] = unit_vector(n);
		vec3 a;
		if (cuAbs(w().x()) > 0.9) {
			a = vec3(0, 1, 0);
		}
		else {
			a = vec3(1, 0, 0);
		}
		axis[1] = unit_vector(cross(w(), a));
		axis[0] = cross(w(), v());
	}

};

class pdf {
public:
	__device__ virtual float value(const vec3& direction) const = 0;
	__device__ virtual vec3 generate(curandState* _s) const = 0;
};

class cosine_pdf : public pdf {
public:
	onb uvw;
public:
	__device__ cosine_pdf(const vec3& w);

	__device__ float value(const vec3& direction) const override;

	__device__ vec3 generate(curandState* _s) const override;

};

class Triangle;
class hittable_pdf : public pdf {
public:
	const Triangle* ptr;
	vec3 o;
public:
	__device__ hittable_pdf(const Triangle* t, const vec3& origin): ptr(t), o(origin) {}

	__device__ float value(const vec3& direction) const override;

	__device__ vec3 generate(curandState* _s) const override;
};

class mixture_pdf : public pdf {
public:
	__device__ mixture_pdf(pdf *_p1, pdf *_p2);
	
	__device__ float value(const vec3& direction) const override;

	__device__ vec3 generate(curandState* _s) const override;

	pdf* p1;
	pdf* p2;
	
};

