#pragma once

#include "Hittable.h"
#include "AABB.h"

class Sphere : public Hittable
{
public:
    Vec3 center;
    float radius;
    Material* mat_ptr;
public:
    __host__ Sphere() {}
    __host__ Sphere(Vec3 cen, float r, Material* m)
        : center(cen), radius(r), mat_ptr(m) {}

    __device__ bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
    __host__ bool BoundingBox(AABB& output_box) const override;
};