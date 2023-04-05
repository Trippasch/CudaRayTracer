#pragma once

#include <cmath>
#include <iostream>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define PI 3.141592654f
#define DEG(rad) rad*57.2957795
#define RAD(deg) deg/57.2957795
#define RND (static_cast<float>(rand()) / static_cast<float>(RAND_MAX))

// #define RND (curand_uniform(&local_rand_state))

class Vec3
{
public:
    __host__ __device__ Vec3() {}
    __host__ __device__ Vec3(const float e0, const float e1, const float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ inline const Vec3& operator+() const { return *this; }
    __host__ __device__ inline Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; };

    __host__ __device__ inline Vec3& operator+=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator-=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator*=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator/=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator*=(const float t);
    __host__ __device__ inline Vec3& operator/=(const float t);

    __host__ __device__ inline float Length() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
    __host__ __device__ inline float SquaredLength() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
    __host__ __device__ inline void MakeUnitVector();

    __host__ __device__ inline void Print() const { printf("[%f, %f, %f]\n", e[0], e[1], e[2]); }

    __host__ __device__ inline bool NearZero() const;

    float e[3];
};

inline std::istream& operator>>(std::istream &is, Vec3 &t) {
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream &os, const Vec3 &t) {
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

__host__ __device__ inline void Vec3::MakeUnitVector() {
    float k = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    e[0] *= k; e[1] *= k; e[2] *= k;
}

__host__ __device__ inline Vec3 operator+(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline Vec3 operator-(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline Vec3 operator/(const Vec3 &v1, const Vec3 &v2) {
    return Vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline Vec3 operator*(float t, const Vec3 &v) {
    return Vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline Vec3 operator/(Vec3 v, float t) {
    return Vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &v, float t) {
    return Vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline float Dot(const Vec3 &v1, const Vec3 &v2) {
    return v1.e[0] *v2.e[0] + v1.e[1] *v2.e[1]  + v1.e[2] *v2.e[2];
}

__host__ __device__ inline Vec3 Cross(const Vec3 &v1, const Vec3 &v2) {
    return Vec3( (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
                (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])),
                (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));
}


__host__ __device__ inline Vec3& Vec3::operator+=(const Vec3 &v){
    e[0]  += v.e[0];
    e[1]  += v.e[1];
    e[2]  += v.e[2];
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const Vec3 &v){
    e[0]  *= v.e[0];
    e[1]  *= v.e[1];
    e[2]  *= v.e[2];
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const Vec3 &v){
    e[0]  /= v.e[0];
    e[1]  /= v.e[1];
    e[2]  /= v.e[2];
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator-=(const Vec3& v) {
    e[0]  -= v.e[0];
    e[1]  -= v.e[1];
    e[2]  -= v.e[2];
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const float t) {
    e[0]  *= t;
    e[1]  *= t;
    e[2]  *= t;
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const float t) {
    float k = 1.0/t;

    e[0]  *= k;
    e[1]  *= k;
    e[2]  *= k;
    return *this;
}

__host__ __device__ inline bool Vec3::NearZero() const {
    // Return true if the vector is close to zero in all dimensions.
    const auto s = 1e-8;
    return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
}

__host__ __device__ inline Vec3 UnitVector(Vec3 v) {
    return v / v.Length();
}

__host__ __device__ inline Vec3 Normalize(Vec3 v)
{
    float invLen = 1 / sqrtf(Dot(v, v));
    return invLen * v;
}

__device__ inline Vec3 Random(curandState* local_rand_state)
{
    return Vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state));
}

__device__ inline int RandomAxis(curandState* local_rand_state)
{
    return static_cast<int>(3*curand_uniform(local_rand_state));
}

__host__ inline int RandomIntRange(int min, int max)
{
    int range = max - min + 1;
    return rand() % range + min;
}

__host__ __device__ inline float LengthSquared(Vec3 e)
{
    return e.x() * e.x() + e.y() * e.y() + e.z() * e.z();
}

// // clamp x to range [a, b]
// __device__ inline float Clamp(float x, float a, float b)
// {
//     return max(a, min(b, x));
// }

// __device__ inline int Clamp(int x, int a, int b)
// {
//     return max(a, min(b, x));
// }

__device__ inline Vec3 RandomInUnitSphere(curandState* local_rand_state)
{
    Vec3 p;
    do
    {
        p = 2.0f * Random(local_rand_state) - Vec3(1.0f, 1.0f, 1.0f);
    } while (LengthSquared(p) >= 1.0f);
    return p;
}

__device__ inline Vec3 RandomInUnitVector(curandState* local_rand_state)
{
    return UnitVector(RandomInUnitSphere(local_rand_state));
}

__device__ inline Vec3 RandomInHemisphere(const Vec3& normal, curandState* local_rand_state)
{
    Vec3 in_unit_sphere = RandomInUnitSphere(local_rand_state);
    if (Dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

__device__ inline Vec3 RandomInUnitDisk(curandState* local_rand_state) {
    Vec3 p;
    do {
        p = 2.0f * Vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0.0f) - Vec3(1.0f, 1.0f, 0.0f);
    } while (Dot(p, p) >= 1.0f);
    return p;
}

__host__ __device__ inline Vec3 Reflect(const Vec3& v, const Vec3& n) {
    return v - 2.0f * Dot(v, n) * n;
}

__host__ __device__ inline bool Refract(const Vec3& v, const Vec3& n, float eati_over_etat, Vec3& refracted) {
    Vec3 uv = UnitVector(v);
    float dt = Dot(uv, n);
    float discriminant = 1.0f - eati_over_etat * eati_over_etat * (1 - dt * dt);
    if (discriminant > 0)
    {
        refracted = eati_over_etat * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }

    return false;
}