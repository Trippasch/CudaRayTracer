#pragma once

#include "AABB.h"

class Material;

typedef struct HitRecord
{
    Vec3 p;
    Vec3 normal;
    Material* mat_ptr;
    float t;
    bool front_face;

    __device__ inline void SetFaceNormal(const Ray& r, const Vec3& outward_normal)
    {
        front_face = Dot(r.Direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
} HitRecord;

typedef enum HittableType
{
    HITTABLE_LIST = 0,
    BVH_NODE,
    SPHERE
} HittableType;

class Hittable
{
public:
    HittableType type;

    __host__ bool HittableBoundingBox(AABB &box, Hittable* hittable);
};