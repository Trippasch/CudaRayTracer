#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define PI 3.141592654f
#define DEG(rad) rad*57.2957795
#define RAD(deg) deg/57.2957795

#define RND (curand_uniform(&local_rand_state))

inline __host__ __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator+(const float3& a, const float& b) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ float3 operator/(const float3& a, const float3& b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline __host__ __device__ float3 operator*(const float& a, const float3& b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

inline __host__ __device__ float3 operator*(const float3& b, const float& a) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

inline __host__ __device__ float3 operator/(const float3& b, const float& a) {
    return make_float3(b.x / a, b.y / a, b.z / a);
}

inline __host__ __device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator-(const float3& v) {
    return make_float3(-v.x, -v.y, -v.z);
}

inline __host__ __device__ float3 fromShort(const short3& a) {
    return make_float3(a.x, a.y, a.z);
}


// shorties
inline __host__ __device__ short3 operator+(const short3& a, const short3& b) {
    return make_short3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ short3 operator+(const short3& a, const short& b) {
    return make_short3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ short3 operator*(const short3& a, const short3& b) {
    return make_short3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ short3 operator/(const short3& a, const short3& b) {
    return make_short3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline __host__ __device__ short3 operator*(const short& a, const short3& b) {
    return make_short3(a * b.x, a * b.y, a * b.z);
}

inline __host__ __device__ short3 operator*(const short3& b, const short& a) {
    return make_short3(a * b.x, a * b.y, a * b.z);
}

inline __host__ __device__ short3 operator-(const short3& a, const short3& b) {
    return make_short3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ short3 fromLong(const float3& a) {
    return make_short3(a.x, a.y, a.z);
}


inline __host__ __device__ float3 floor(const float3& a) {
    return make_float3(floor(a.x), floor(a.y), floor(a.z));
}

inline __host__ __device__ float2 floor(const float2& a) {
    return make_float2(floor(a.x), floor(a.y));
}

inline __host__ __device__ float2 operator*(const float& a, const float2& b) {
    return make_float2(a * b.x, a * b.y);
}

inline __host__ __device__ float2 operator+(const float2& a, const float2& b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ float2 operator*(const float2& a, const float2& b) {
    return make_float2(a.x * b.x, a.y * b.y);
}


inline __host__ __device__  float dot(float3 v1, float3 v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

inline __host__ __device__  float dot(float2 v1, float2 v2)
{
    return v1.x * v2.x + v1.y * v2.y;
}

inline __host__ __device__  float3 cross(float3 v1, float3 v2)
{
    return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

inline __host__ __device__ float length(float3 v)
{
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline __host__ __device__ float length1(float3 v)
{
    return v.x + v.y + v.z;
}

inline __host__ __device__ float3 inverse(float3 v)
{
    return make_float3(-v.x, -v.y, -v.z);
}

inline __host__ __device__ float3 normalize(float3 v)
{
    float invLen = 1 / sqrtf(dot(v, v));
    return invLen * v;
}

inline __host__ __device__ float3 unit_vector(float3 v) {
    return v / length(v);
}

inline __device__ float3 random(curandState* local_rand_state)
{
    return make_float3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state));
}

inline __host__ __device__ float length_squared(float3 e)
{
    return e.x * e.x + e.y * e.y + e.z * e.z;
}

inline __device__ float3 random_in_unit_sphere(curandState* local_rand_state)
{
    float3 p;
    do
    {
        p = 2.0f * random(local_rand_state) - make_float3(1, 1, 1);
    } while (length_squared(p) >= 1.0f);
    return p;
}

inline __device__ float3 random_in_unit_disk(curandState* local_rand_state) {
    float3 p;
    do {
        p = 2.0f * make_float3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - make_float3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}

inline __host__ __device__ float3 reflect(const float3& v, const float3& n) {
    return v - 2.0f * dot(v, n) * n;
}

inline __host__ __device__ bool refract(const float3& v, const float3& n, float eati_over_etat, float3& refracted) {
    float3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - eati_over_etat * eati_over_etat * (1 - dt * dt);
    if (discriminant > 0)
    {
        refracted = eati_over_etat * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }

    return false;
}
