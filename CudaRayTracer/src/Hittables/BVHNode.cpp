#include "BVHNode.h"

#include "Utils/helper_cuda.h"

#include "Core/Log.h"

__host__ bool BVHNode::BoundingBox(AABB &box) const
{
    box = this->box;
    return true;
}

__host__ inline bool BoxCompare(const Sphere* a, const Sphere* b, int axis)
{
    AABB box_a;
    AABB box_b;

    if (!a->BoundingBox(box_a) || !b->BoundingBox(box_b))
        RT_TRACE("No bounding box in BVHNode constructor.");

    return box_a.Min().e[axis] < box_b.Min().e[axis];
}

__host__ inline bool BoxXCompare(const Sphere* a, const Sphere* b)
{
    return BoxCompare(a, b, 0);
}

__host__ inline bool BoxYCompare(const Sphere* a, const Sphere* b)
{
    return BoxCompare(a, b, 1);
}

__host__ inline bool BoxZCompare(const Sphere* a, const Sphere* b)
{
    return BoxCompare(a, b, 2);
}

__host__ BVHNode::BVHNode(std::vector<Sphere*> list, size_t start, size_t end)
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
        std::sort(objects.begin() + start, objects.begin() + end, comparator);

        auto mid = start + object_span / 2;

        checkCudaErrors(cudaMallocManaged((void**)&left, sizeof(Sphere )));
        checkCudaErrors(cudaMallocManaged((void**)&right, sizeof(Sphere )));
        left = new BVHNode(objects, start, mid);
        right = new BVHNode(objects, mid, end);
    }

    AABB box_left, box_right;

    if (!left->BoundingBox(box_left) || !right->BoundingBox(box_right)) {
        RT_TRACE("No bounding box in BVHNode constructor.");
    }

    box = SurroundingBox(box_left, box_right);
}