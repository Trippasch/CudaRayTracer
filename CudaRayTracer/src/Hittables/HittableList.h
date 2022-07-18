#pragma once

#include "Hittable.h"

#include <memory>
#include <vector>

class hittable_list : public hittable
{
public:
    hittable** list;
    int size;
public:
    __device__ hittable_list() {}
    __device__ hittable_list(hittable** l, int n) { list = l, size = n; }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
};

__device__ bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < size; i++)
    {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}
