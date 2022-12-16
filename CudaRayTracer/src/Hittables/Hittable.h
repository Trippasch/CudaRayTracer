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

class Hittable
{
public:
    __device__ virtual bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
    __host__ virtual bool BoundingBox(AABB& output_box) const = 0;
    __host__ virtual Hittable* Clone() const = 0;
};
