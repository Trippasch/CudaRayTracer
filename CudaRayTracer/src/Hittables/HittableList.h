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
    __device__ HittableList() {}
    __device__ HittableList(Hittable** l, int n) { list = l, size = n; }

    __device__ bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
    __device__ bool BoundingBox(float time0, float time1, AABB& output_box) const override;
};

__device__ bool HittableList::Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const
{
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < size; i++) {
        if (list[i]->Hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

__device__ bool HittableList::BoundingBox(float time0, float time1, AABB& output_box) const
{
    if (size < 1)
        return false;

    AABB temp_box;
    bool first_box = true;

    for (int i = 0; i < size; i++) {
        if (!list[i]->BoundingBox(time0, time1, temp_box))
            return false;
        // output_box = first_box ? temp_box : SurroundingBox(output_box, temp_box);
        first_box = false;
    }

    return true;
}