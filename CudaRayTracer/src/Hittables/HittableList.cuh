// #pragma once

// #include <vector>

// #include "Hittables/Hittable.cuh"

// class HittableList
// {
// public:
//     std::vector<Hittable *> objects;

// public:
//     __host__ HittableList() {}
//     __host__ HittableList(Hittable *object) { Add(object); }

//     __host__ inline void Add(Hittable *object)
//     {
//         objects.push_back(object);
//     }

//     __host__ inline void Clear()
//     {
//         objects.clear();
//     }

//     // __device__ inline bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const
//     // {
//     //     HitRecord temp_rec;
//     //     bool hit_anything = false;
//     //     float closest_so_far = t_max;

//     //     for (int i = 0; i < size; i++) {
//     //         if (list[i]->Hit(r, t_min, closest_so_far, temp_rec)) {
//     //             hit_anything = true;
//     //             closest_so_far = temp_rec.t;
//     //             rec = temp_rec;
//     //         }
//     //     }

//     //     return hit_anything;
//     // }


//     // __host__ inline bool BoundingBox(AABB& output_box) const
//     // {
//     //     if (objects.empty()) return false;

//     //     AABB temp_box;
//     //     bool first_box = true;

//     //     for (const auto& object : objects) {
//     //         if (!object->HittableBoundingBox(temp_box, object)) return false;
//     //         output_box = first_box ? temp_box : SurroundingBox(output_box, temp_box);
//     //         first_box = false;
//     //     }

//     //     return true;
//     // }
// };