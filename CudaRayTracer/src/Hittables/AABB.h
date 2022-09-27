#pragma once

#include "../Utils/Ray.h"

#include <thrust/swap.h>

class AABB {
public:
    Vec3 minimum;
    Vec3 maximum;
public:
    __device__ AABB() {}
    __device__ AABB(const Vec3 &a, const Vec3 &b) { minimum = a; maximum = b; }

    __device__ Vec3 Min() const { return minimum; }
    __device__ Vec3 Max() const { return maximum; }

    __device__ inline bool Hit(const Ray &r, float t_min, float t_max) const {
        for (int a = 0; a < 3; a++) {
            auto invD = 1.0f / r.Direction()[a];
            auto t0 = (Min()[a] - r.Origin()[a]) * invD;
            auto t1 = (Max()[a] - r.Origin()[a]) * invD;
            if (invD < 0.0f)
                thrust::swap(t0, t1);
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if (t_max <= t_min)
                return false;
        }
        return true;
    }
};

// __device__ AABB SurroundingBox(AABB box0, AABB box1) {
//     Vec3 small = (
//     std::min(box0.Min().x(), box1.Min().x()),
//     std::min(box0.Min().y(), box1.Min().y()),
//     std::min(box0.Min().z(), box1.Min().z()));

//     Vec3 big = (
//     std::max(box0.Max().x(), box1.Max().x()),
//     std::max(box0.Max().y(), box1.Max().y()),
//     std::max(box0.Max().z(), box1.Max().z()));

//     return AABB(small, big);
// }