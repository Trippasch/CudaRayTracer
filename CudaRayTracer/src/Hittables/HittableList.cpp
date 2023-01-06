#include "HittableList.h"

// __device__ bool HittableList::Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const
// {
//     HitRecord temp_rec;
//     bool hit_anything = false;
//     float closest_so_far = t_max;

//     for (int i = 0; i < objects.size(); i++) {
//         if (objects[i]->Hit(r, t_min, closest_so_far, temp_rec)) {
//             hit_anything = true;
//             closest_so_far = temp_rec.t;
//             rec = temp_rec;
//         }
//     }

//     return hit_anything;
// }

// __host__ bool HittableList::BoundingBox(AABB& output_box) const
// {
//     if (objects.size() < 1)
//         return false;

//     AABB temp_box;
//     bool first_box = true;

//     for (int i = 0; i < objects.size(); i++) {
//         if (!objects[i]->BoundingBox(temp_box))
//             return false;
//         output_box = first_box ? temp_box : SurroundingBox(output_box, temp_box);
//         first_box = false;
//     }

//     return true;
// }