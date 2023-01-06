#pragma once

#include "Sphere.h"

#include <memory>
#include <vector>

class HittableList
{
public:
    std::vector<Sphere*> objects;
public:
    __host__ HittableList() {}
    __host__ HittableList(Sphere* object) { Add(object); }

    __host__ void Add(Sphere* object)
    {
        objects.push_back(object);
    }

    __host__ void Clear() { objects.clear(); }


    // __device__ bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
    // __host__ bool BoundingBox(AABB& output_box) const override;
    // __host__ inline HittableList* Clone() const override { return new HittableList(*this); }
};