#pragma once

#include "Utils/Vec3.h"

class ray
{
private:
    float3 orig;
    float3 dir;
public:
    __device__ ray() : orig(make_float3(0.0f, 0.0f, 0.0f)), dir(make_float3(0.0f, 0.0f, 0.0f)) {}
    __device__ ray(const float3& origin, const float3& direction) : orig(origin), dir(direction) {}
    __device__ inline float3 origin() const { return orig; }
    __device__ inline float3 direction() const { return dir; }
    __device__ inline float3 point_at_parameter(float t) const { return orig + (t * dir); }
};