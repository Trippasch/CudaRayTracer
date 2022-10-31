#pragma once

#include "HittableList.h"

#include <vector>
#include <memory>

class BVHNode : public Hittable
{
public:
    AABB box;
    Hittable* left;
    Hittable* right;
public:
    __host__ BVHNode();
    __host__ BVHNode(Hittable** list, size_t size)
        : BVHNode(list, 0, size) {}
    __host__ BVHNode(Hittable** list, size_t start, size_t end);
    __device__ bool Hit(const Ray &ray, float tmin, float tmax, HitRecord &rec) const override;
    __host__ bool BoundingBox(AABB &box) const override;
};