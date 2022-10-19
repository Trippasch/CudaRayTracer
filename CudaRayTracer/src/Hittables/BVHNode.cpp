#include "BVHNode.h"

#include <iostream>

#include <thrust/sort.h>
#include <thrust/functional.h>

__device__ bool BVHNode::Hit(const Ray &ray, float tmin, float tmax, HitRecord &rec) const
{
    if (!box.Hit(ray, tmin, tmax))
        return false;

    bool hit_left = left->Hit(ray, tmin, tmax, rec);
    bool hit_right = right->Hit(ray, tmin, hit_left ? rec.t : tmax, rec);

    return hit_left || hit_right;
}

__host__ bool BVHNode::BoundingBox(AABB &box) const
{
    box = this->box;
    return true;
}

__host__ inline bool BoxCompare(const Hittable* a, const Hittable* b, int axis)
{
    AABB box_a;
    AABB box_b;

    return box_a.Min().e[axis] < box_b.Min().e[axis];
}

__host__ inline bool BoxXCompare(const Hittable* a, const Hittable* b)
{
    return BoxCompare(a, b, 0);
}

__host__ inline bool BoxYCompare(const Hittable* a, const Hittable* b)
{
    return BoxCompare(a, b, 1);
}

__host__ inline bool BoxZCompare(const Hittable* a, const Hittable* b)
{
    return BoxCompare(a, b, 2);
}

__host__ BVHNode::BVHNode(Hittable** list, size_t start, size_t end)
{
    auto objects = list;

    int axis = RandomIntRange(0, 2);
    auto comparator = (axis == 0) ? BoxXCompare
                    : (axis == 1) ? BoxYCompare
                                  : BoxZCompare;

    size_t object_span = end - start;

    if (object_span == 1) {
        left = right = objects[start];
    }
    else if (object_span == 2) {
        if (comparator(objects[start], objects[start + 1])) {
            left = objects[start];
            right = objects[start + 1];
        }
        else {
            left = objects[start + 1];
            right = objects[start];
        }
    }
    else {

        auto mid = start + object_span / 2;

        cudaMallocManaged((void**)&left, sizeof(Hittable ));
        cudaMallocManaged((void**)&right, sizeof(Hittable ));
        left = new BVHNode(objects, start, mid);
        right = new BVHNode(objects, mid, end);
    }

    AABB box_left, box_right;

    if (left != nullptr && right != nullptr) {
        if (!left->BoundingBox(box_left) || !right->BoundingBox(box_right)) {
            printf("No bounding box in BVHNode constructor.\n");
        }
    }

    box = SurroundingBox(box_left, box_right);
}