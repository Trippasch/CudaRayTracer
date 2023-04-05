#pragma once

#include "Math.cuh"

class Ray
{
private:
    Vec3 orig;
    Vec3 dir;
public:
    __device__ Ray() : orig(0.0f, 0.0f, 0.0f), dir(0.0f, 0.0f, 0.0f) {}
    __device__ Ray(const Vec3& origin, const Vec3& direction) : orig(origin), dir(direction) {}
    __device__ inline Vec3 Origin() const { return orig; }
    __device__ inline Vec3 Direction() const { return dir; }
    __device__ inline Vec3 PointAtParameter(float t) const { return orig + (t * dir); }
};