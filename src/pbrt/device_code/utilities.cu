#include "utilities.h"
#include "triangle.h"

__device__ void getAttenuationFromTexture(cudaTextureObject_t *_t, float _u, float _v, int _texIdx, vec3& _attenu) {
    float4 tem = tex2D<float4>(_t[_texIdx], _u, _v);
    _attenu[0] = tem.x;
    _attenu[1] = tem.y;
    _attenu[2] = tem.z;
}

__device__ void getAttenuationFromTexture(CudaTexture* _t, float _u, float _v, int _texIdx, vec3& _attenu) {
    RGB tem = _t[_texIdx].fetch(_u, _v);
    _attenu[0] = tem.r;
    _attenu[1] = tem.g;
    _attenu[2] = tem.b;
}

__device__ vec3 random_cosine_direction(curandState* _s) {
    float r1 = CURAND(_s);
    float r2 = CURAND(_s);
    float z = sqrt(1 - r2);
    float phi = 2 * M_PI * r1;
    float x = __cosf(phi) * 2.0 * sqrt(r2);
    float y = __sinf(phi) * 2.0 * sqrt(r2);
    return vec3(x, y, z);
}

__device__ vec3 random_to_sphere(float r, float d2, curandState* _s) {
    float r1 = CURAND(_s);
    float r2 = CURAND(_s);
    float z = 1 + r2 * (sqrt(1 - r * r / d2) - 1);
    float phi = 2 * M_PI * r1;
    float x = __cosf(phi) * sqrt(1 - z * z);
    float y = __sinf(phi) * sqrt(1 - z * z);
    return vec3(x, y, z);
}


__device__ float hittable_pdf::value(const vec3& direction) const  {
    return ptr->pdf_value(o, direction);
}

__device__ vec3 hittable_pdf::generate(curandState* _s) const  {
    return ptr->random(o, _s);
}


__device__ cosine_pdf::cosine_pdf(const vec3& w) {
    uvw.build_from_w(w);
}

__device__ float cosine_pdf::value(const vec3& direction) const  {
    float cosine = dot(unit_vector(direction), uvw.w());
    return cosine > 0 ? cosine / M_PI : 0;
}

__device__ vec3 cosine_pdf::generate(curandState* _s) const  {
    return uvw.local(random_cosine_direction(_s));
}


__device__ mixture_pdf::mixture_pdf(pdf *_p1, pdf *_p2): p1(_p1), p2(_p2) {

}

__device__ float mixture_pdf::value(const vec3& direction) const{
    return 0.5 * p1->value(direction) + 0.5 * p2->value(direction);
}

__device__ vec3 mixture_pdf::generate(curandState* _s) const {
    if (CURAND(_s) < 0.5) {
        return p1->generate(_s);
    }
    else {
        return p2->generate(_s);
    }
}


