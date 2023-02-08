#include "Hittable.h"

#include "Sphere.h"
#include "BVHNode.h"
#include "HittableList.h"

__host__ bool Hittable::HittableBoundingBox(AABB &box, Hittable* hittable)
{
    switch (hittable->type)
    {
    case SPHERE:
    {
        Sphere* sphere = static_cast<Sphere *>(hittable);
        return sphere->BoundingBox(box);
    }
        break;
    case BVH_NODE:
    {
        BVHNode* node = static_cast<BVHNode *>(hittable);
        return node->BoundingBox(box);
    }
        break;
    case HITTABLE_LIST:
    {
        HittableList* list = static_cast<HittableList *>(hittable);
        return list->BoundingBox(box);
    }
        break;
    default:
        break;
    }

    return false;
}