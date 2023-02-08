// #pragma once

// #include "Hittable.h"
// #include "AABB.h"

// #include "Sphere.h"

// #include <vector>
// #include <memory>

// class BVHNode : public Hittable
// {
// public:
//     AABB box;
//     Hittable* left;
//     Hittable* right;
// public:
//     __host__ BVHNode();
//     __host__ BVHNode(std::vector<Hittable*> list)
//         : BVHNode(list, 0, list.size()) {}
//     __host__ BVHNode(std::vector<Hittable*> list, size_t start, size_t end);

//     __host__ bool BVHNode::BoundingBox(AABB &box) const;

//     __device__ inline bool BVHNode::Hit(const Ray &ray, float tmin, float tmax, HitRecord &rec) const
//     {
//         if (!box.Hit(ray, tmin, tmax))
//             return false;

//         // bool hit_left = left->HittableHit(ray, tmin, tmax, rec, left);
//         bool hit_left = false;

//         switch (left->type)
//         {
//         case SPHERE:
//         {
//             Sphere* sphere = static_cast<Sphere *>(left);
//             hit_left = sphere->Hit(ray, tmin, tmax, rec);
//         }
//             break;
//         case BVH_NODE:
//         {
//             BVHNode* node = static_cast<BVHNode *>(left);
//             hit_left = node->Hit(ray, tmin, tmax, rec);
//         }
//             break;
//         // case HITTABLE_LIST:
//         // {
//         //     HittableList* list = static_cast<HittableList *>(hittable);
//         //     return list->Hit(ray, tmin, tmax, rec);
//         // }
//         //     break;
//         default:
//             break;
//         }

//         // bool hit_right = right->HittableHit(ray, tmin, hit_left ? rec.t : tmax, rec, right);
//         bool hit_right = false;

//         switch (right->type)
//         {
//         case SPHERE:
//         {
//             Sphere* sphere = static_cast<Sphere *>(right);
//             hit_right = sphere->Hit(ray, tmin, tmax, rec);
//         }
//             break;
//         case BVH_NODE:
//         {
//             BVHNode* node = static_cast<BVHNode *>(right);
//             hit_right = node->Hit(ray, tmin, tmax, rec);
//         }
//             break;
//         // case HITTABLE_LIST:
//         // {
//         //     HittableList* list = static_cast<HittableList *>(hittable);
//         //     return list->Hit(ray, tmin, tmax, rec);
//         // }
//         //     break;
//         default:
//             break;
//         }

//         return hit_left || hit_right;
//     }
// };