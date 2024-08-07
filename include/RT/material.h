#pragma once
#include "ray.h"
#include "cuda_runtime.h"
#include "utilities.h"
#include "texture.h"

__device__ inline float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ inline bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ inline vec3 random_in_unit_sphere(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

class hit_record {
public:
    __device__ hit_record() {}
    float t;
    vec3 p;
    vec3 normal;
    int matType = 0;
    int state = 1; // 0 is valid, 1 is invalid 
    float u = 0.0;
    float v = 0.0;
    int texIdx = 0;
    bool hitFront = true;

    __host__ __device__ bool isValid() {
        return state == 0;
    }

    __host__ __device__ void reset() {
        state = 1;
    }
};

struct scatter_record {
    ray specular_ray;
    bool is_specular;
    vec3 attenuation;
    pdf* pdf_ptr;
};

class material {
public:
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, scatter_record& srec, curandState* s, 
            TEX_OBJ _textures) const = 0;

    __device__ virtual float scatter_pdf(const ray& r_in, const hit_record& rec, ray& scattered) const {
        return 1.0;
    }
    __device__ virtual vec3 emitted(float u=0.0f, float v = 0.0f)  const {
        return vec3();
    }

    __device__ virtual vec3 getAlbedo() const {
        return vec3(1.0, 1.0, 1.0);
    }
};


class lambertian : public material {
public:
    __device__ lambertian(){

    }
    __device__ lambertian(const vec3& a) : albedo(a) {
    
    }

    __device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, scatter_record& srec, curandState* s,
        TEX_OBJ _textures) const override
    {
        srec.is_specular = false;
        srec.pdf_ptr = new cosine_pdf(rec.normal);
        if (rec.texIdx == -1) {
            srec.attenuation = albedo;
        }
        //
        else {
            getAttenuationFromTexture(_textures, rec.u, rec.v, rec.texIdx, srec.attenuation);
        }
        
        return true;
    }

    __device__ float scatter_pdf(const ray& r_in, const hit_record& rec, ray& scattered) const override {
        //return 1.0;
        float cosine = dot(rec.normal, unit_vector(scattered.direction()));
        if (cosine < 0.0) cosine = 0.0;
        return cosine / 3.1415926;
    }
    /*__device__ virtual vec3 emitted(float u = 0, float v = 0.0f)  const {
        return albedo;
    }*/

    __device__ vec3 getAlbedo() const override{
        return albedo;
    }

    vec3 albedo;
};

class diffusive_light : public material {
public:
    __device__ diffusive_light() {

    }
    __device__ diffusive_light(const vec3& a) : albedo(a) {

    }

    __device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, scatter_record& srec, curandState* s,
        TEX_OBJ _textures) const override {

        if (rec.texIdx == -1) {
            srec.attenuation = albedo;
        }
        //
        else {
            getAttenuationFromTexture(_textures, rec.u, rec.v, rec.texIdx, srec.attenuation);
        }
        return false;
    }

    __device__ virtual vec3 emitted(float u = 0, float v = 0.0f)  const {
        return albedo;
    }

    vec3 albedo;
};


class metal : public material {
public:
    __device__ metal(const vec3& a, float f) : albedo(a) {
        fuzz = f < 1.0 ? f : 1.0;
    }
    __device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, scatter_record& srec, curandState* s,
        TEX_OBJ _textures) const override {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        srec.specular_ray = ray(rec.p, reflected + fuzz * random_in_unit_sphere(s), r_in.time());
        srec.attenuation = albedo;
        srec.is_specular = true;
        srec.pdf_ptr = nullptr;
        return (dot(srec.specular_ray.direction(), rec.normal) > 0.0f);
    }

    vec3 albedo = vec3(0.2f, 0.3f, 0.8f);
    float fuzz = 0.2f;
};

class dielectric : public material {
public:
    __device__ dielectric(float ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(const ray& r_in,
        const hit_record& rec,
        vec3& attenuation,
        scatter_record& srec,
        curandState* local_rand_state, TEX_OBJ _textures) const {
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        srec.attenuation = vec3(1.0, 1.0, 1.0);
        vec3 refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0f;
        if (curand_uniform(local_rand_state) < reflect_prob)
            srec.specular_ray = ray(rec.p, reflected, r_in.time());
        else
            srec.specular_ray = ray(rec.p, refracted, r_in.time());
        srec.is_specular = true;
        srec.pdf_ptr = nullptr;
       
        return true;
    }

    float ref_idx;
};