#pragma once

#include "Sphere.h"

#include <vector>
#include <memory>

class BVHNode : public Sphere
{
public:
    AABB box;
    Sphere* left;
    Sphere* right;
public:
    __host__ BVHNode();
    __host__ BVHNode(std::vector<Sphere*> list)
        : BVHNode(list, 0, list.size()) {}
    __host__ BVHNode(std::vector<Sphere*> list, size_t start, size_t end);

    __host__ bool BVHNode::BoundingBox(AABB &box) const;

    __device__ inline bool BVHNode::Hit(const Ray &ray, float tmin, float tmax, HitRecord &rec) const
    {
        if (!box.Hit(ray, tmin, tmax))
            return false;

        bool hit_left = left->Hit(ray, tmin, tmax, rec);
        bool hit_right = right->Hit(ray, tmin, hit_left ? rec.t : tmax, rec);

        return hit_left || hit_right;
    }
};