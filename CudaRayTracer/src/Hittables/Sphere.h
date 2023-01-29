#pragma once

#include "Material.h"

class Sphere
{
public:
    Vec3 center;
    float radius;
    Material* mat_ptr;
public:
    __host__ __device__ Sphere() {}
    __host__ __device__ Sphere(Vec3 cen, float r, Material* m)
        : center(cen), radius(r), mat_ptr(m) {}

    __device__ inline bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const
    {
        Vec3 oc = r.Origin() - center;
        float a = Dot(r.Direction(), r.Direction());
        float b = Dot(oc, r.Direction());
        float c = Dot(oc, oc) - radius*radius;

        float discriminant = b*b - a*c;
        if (discriminant > 0) {
            float temp = (-b - sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.p = r.PointAtParameter(rec.t);
                rec.normal = (rec.p - center) / radius;
                rec.mat_ptr = mat_ptr;
                return true;
            }
            temp = (-b + sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                rec.t = temp;
                rec.p = r.PointAtParameter(rec.t);
                rec.normal = (rec.p - center) / radius;
                rec.mat_ptr = mat_ptr;
                return true;
            }
        }

        return false;
    }

    __host__ inline bool Sphere::BoundingBox(AABB& output_box) const
    {
        output_box = AABB(
            center - Vec3(radius, radius, radius),
            center + Vec3(radius, radius, radius));
        return true;
    }
};