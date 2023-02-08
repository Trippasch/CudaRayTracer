#pragma once

#include <vector>

#include "Sphere.h"

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

    // __device__ inline bool Hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const {
    //     HitRecord temp_rec;
    //     bool hit_anything = false;
    //     float closest_so_far = t_max;

    //     for (int i = 0; i < 5; i++) {
    //         if (object->HittableHit(r, t_min, closest_so_far, temp_rec, object)) {
    //             hit_anything = true;
    //             closest_so_far = temp_rec.t;
    //             rec = temp_rec;
    //         }
    //     }

    //     return hit_anything;
    // }


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