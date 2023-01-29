#pragma once

#include "../Utils/Ray.h"

#include <thrust/swap.h>
#include <thrust/extrema.h>

class AABB {
public:
    Vec3 minimum;
    Vec3 maximum;
public:
    __host__ AABB() : minimum(Vec3(0.0f, 0.0f ,0.0f)), maximum(Vec3(0.0f, 0.0f ,0.0f)) {}
    __host__ AABB(const Vec3 &a, const Vec3 &b) { minimum = a; maximum = b; }

    __host__ __device__ Vec3 Min() const { return minimum; }
    __host__ __device__ Vec3 Max() const { return maximum; }

    __device__ inline bool Hit(const Ray &r, float t_min, float t_max) const {
        for (int a = 0; a < 3; a++) {
            auto invD = 1.0f / r.Direction()[a];
            auto t0 = (Min()[a] - r.Origin()[a]) * invD;
            auto t1 = (Max()[a] - r.Origin()[a]) * invD;
            if (invD < 0.0f) {
                // thrust::swap(t0, t1);
                auto temp = t0;
                t0 = t1;
                t1 = temp;
            }
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if (t_max <= t_min)
                return false;
        }
        return true;
    }
};

__forceinline__ __host__ AABB SurroundingBox(AABB box0, AABB box1) {
    Vec3 small(
        fmin(box0.Min().x(), box1.Min().x()),
        fmin(box0.Min().y(), box1.Min().y()),
        fmin(box0.Min().z(), box1.Min().z())
    );

    Vec3 big(
        fmax(box0.Max().x(), box1.Max().x()),
        fmax(box0.Max().y(), box1.Max().y()),
        fmax(box0.Max().z(), box1.Max().z())
    );

    return AABB(small, big);
}