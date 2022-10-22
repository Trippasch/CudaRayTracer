#pragma once

#include "Hittable.h"

#include <memory>
#include <vector>

class HittableList : public Hittable
{
public:
    std::vector<Hittable*> objects;
public:
    __host__ HittableList() {}
    __host__ HittableList(Hittable* object) { Add(object); }

    __host__ void Add(Hittable* object)
    {
        objects.push_back(object);
    }

    __host__ void Clear() { objects.clear(); }


    __device__ bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
    __host__ bool BoundingBox(AABB& output_box) const override;
};