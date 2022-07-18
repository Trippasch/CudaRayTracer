#pragma once

#include "../Utils/Ray.h"

class material;

typedef struct hit_record
{
    float3 p;
    float3 normal;
    material* mat_ptr;
    float t;
    bool front_face;

    __device__ inline void set_face_normal(const ray& r, const float3& outward_normal)
    {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
} hit_record;

class hittable
{
public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};
