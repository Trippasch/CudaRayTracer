#pragma once

#include <vector>

#include "Sphere.h"

class HittableList : public Sphere
{
public:
    Sphere** list;
    unsigned int size;
public:
    __device__ HittableList() {}
    __device__ HittableList(Sphere** l, unsigned int n) { list = l; size = n; }

    __device__ inline bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const
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


    // __host__ inline bool BoundingBox(AABB& output_box) const
    // {
    //     if (objects.empty()) return false;

    //     AABB temp_box;
    //     bool first_box = true;

    //     for (const auto& object : objects) {
    //         if (!object->HittableBoundingBox(temp_box, object)) return false;
    //         output_box = first_box ? temp_box : SurroundingBox(output_box, temp_box);
    //         first_box = false;
    //     }

    //     return true;
    // }
};