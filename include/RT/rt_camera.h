#pragma once
#include "vec3.h"
#include "ray.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline __device__ vec3 random_in_unit_disk_d(curandState* _s) {
	vec3 p;
	do {
		p = 2.0f * vec3(curand_uniform(_s), curand_uniform(_s), 0) - vec3(1, 1, 0);
	} while (dot(p, p) >= 1.0f);
	return p;
}

class camera {
public:
	__device__ camera() {}
	__device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect,
		float aperture, float focus_dist, float t0, float t1) {
		time0 = t0;
		time1 = t1;
		lens_radius = aperture / 2.0f;
		float theta = (vfov / 180.f) * (float)M_PI;
		float half_height = tan(theta / 2.0f);
		float half_width = aspect * half_height;
		origin = lookfrom;
		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);
		lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
		horizontal = 2.0f * half_width * focus_dist * u;
		vertical = 2.0f * half_height * focus_dist * v;
	}

	__device__ void update(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect,
		float aperture, float focus_dist, float t0, float t1) {
		time0 = t0;
		time1 = t1;
		lens_radius = aperture / 2.0f;
		float theta = (vfov / 180.f) * (float)M_PI;
		float half_height = tan(theta / 2.0f);
		float half_width = aspect * half_height;
		origin = lookfrom;
		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);
		lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
		horizontal = 2.0f * half_width * focus_dist * u;
		vertical = 2.0f * half_height * focus_dist * v;
	}

	__device__ ray get_ray(float s, float t, curandState* _s) {
		vec3 rd = lens_radius * random_in_unit_disk_d(_s);
		vec3 offset = u * rd.x() + v * rd.y();
		//offset = vec3(0, 0, 0);
		auto random_t = time0 + (time1 - time0) * curand_uniform(_s);
		return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset, random_t);
	}

	vec3 origin;
	vec3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	float lens_radius;
	float time0;
	float time1;
};
