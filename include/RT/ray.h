#pragma once
#include "vec3.h"

class ray {
public:
	__device__ __host__  ray() {}
	__device__ __host__  ray(const vec3& a, const vec3& b, float time = 0.0)
	{
		A = a;
		B = b;
		_time = time;
	}
	__device__ __host__ vec3 origin() const noexcept {
		return A;
	}

	__device__ __host__  vec3 direction() const noexcept {
		return B;
	}

	__device__ __host__ vec3 point_at_parameter(float t) const noexcept {
		return A + t * B;
	}

	__device__ __host__ float time() const noexcept {
		return _time;
	}

	vec3 A;
	vec3 B;
	float _time = 1.0;
};