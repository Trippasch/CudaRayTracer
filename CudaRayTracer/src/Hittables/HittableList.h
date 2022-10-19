#pragma once

#include "Hittable.h"

#include <memory>
#include <vector>

class HittableList : public Hittable
{
public:
    Hittable** list;
    int size;
public:
    __host__ HittableList() {}
    __host__ HittableList(Hittable** l, int n) { list = l, size = n; }

    __device__ bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
    __host__ bool BoundingBox(AABB& output_box) const override;
};