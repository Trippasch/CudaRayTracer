#pragma once

#include "Hittable.h"
#include "AABB.h"

class Sphere
{
public:
    Vec3 center;
    float radius;
    Material* mat_ptr;
public:
    __host__ __device__ Sphere() {}
    __host__ __device__ Sphere(Vec3 cen, float r, Material* m)
        : center(cen), radius(r), mat_ptr(m) {}

    __device__ bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const;

    // __host__ bool BoundingBox(AABB& output_box) const override;
    // __host__ inline Sphere* Clone() const override { return new Sphere(*this); }
};