#pragma once

#include "../Utils/Ray.cuh"

class Material;

typedef struct HitRecord
{
    Vec3 p;
    Vec3 normal;
    Material* mat_ptr;
    float t;
    float u, v;
    bool front_face;

    __device__ inline void SetFaceNormal(const Ray& r, const Vec3& outward_normal)
    {
        front_face = Dot(r.Direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
} HitRecord;

enum Hitt {
    sphere = 0,
    xy_rect,
    xz_rect,
    yz_rect
};

class Hittable
{
public:
    Hitt hittable;

    Vec3 center;
    float radius;
    float width;
    float height;
    Material* mat_ptr;

public:
    __host__ Hittable(Vec3 cen, float r, Material* m, Hitt t)
        : center(cen), radius(r), mat_ptr(m), hittable(t) {}

    __host__ Hittable(Vec3 cen, float w, float h, Material* m, Hitt t)
        : center(cen), width(w), height(h), mat_ptr(m), hittable(t) {}

    __device__ inline bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const
    {
        if (hittable == Hitt::sphere) {
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
                    GetSphereUV(rec.normal, rec.u, rec.v);
                    rec.mat_ptr = mat_ptr;
                    return true;
                }
                temp = (-b + sqrt(discriminant)) / a;
                if (temp < t_max && temp > t_min) {
                    rec.t = temp;
                    rec.p = r.PointAtParameter(rec.t);
                    rec.normal = (rec.p - center) / radius;
                    GetSphereUV(rec.normal, rec.u, rec.v);
                    rec.mat_ptr = mat_ptr;
                    return true;
                }
            }

            return false;
        }
        else if (hittable == Hitt::xy_rect) {
            float x0, x1, y0, y1, k;
            x0 = center.x() - (width/2);
            x1 = center.x() + (width/2);
            y0 = center.y() - (height/2);
            y1 = center.y() + (height/2);
            k = center.z();

            float t = (k - r.Origin().z()) / r.Direction().z();

            if (t < t_min || t > t_max)
                return false;

            float x = r.Origin().x() + t*r.Direction().x();
            float y = r.Origin().y() + t*r.Direction().y();
            if (x < x0 || x > x1 || y < y0 || y > y1)
                return false;

            rec.u = (x-x0)/(x1-x0);
            rec.v = (y-y0)/(y1-y0);
            rec.t = t;
            Vec3 outward_normal = Vec3(0.0f, 0.0f, 1.0f);
            rec.SetFaceNormal(r, outward_normal);
            rec.mat_ptr = mat_ptr;
            rec.p = r.PointAtParameter(t);

            return true;
        }
        else if (hittable == Hitt::xz_rect) {
            float x0, x1, z0, z1, k;
            x0 = center.x() - (width/2);
            x1 = center.x() + (width/2);
            z0 = center.z() - (height/2);
            z1 = center.z() + (height/2);
            k = center.y();

            float t = (k - r.Origin().y()) / r.Direction().y();

            if (t < t_min || t > t_max)
                return false;

            float x = r.Origin().x() + t*r.Direction().x();
            float z = r.Origin().z() + t*r.Direction().z();
            if (x < x0 || x > x1 || z < z0 || z > z1)
                return false;

            rec.u = (x-x0)/(x1-x0);
            rec.v = (z-z0)/(z1-z0);
            rec.t = t;
            Vec3 outward_normal = Vec3(0.0f, 1.0f, 0.0f);
            rec.SetFaceNormal(r, outward_normal);
            rec.mat_ptr = mat_ptr;
            rec.p = r.PointAtParameter(t);

            return true;
        }
        else if (hittable == Hitt::yz_rect) {
            float y0, y1, z0, z1, k;
            y0 = center.y() - (height/2);
            y1 = center.y() + (height/2);
            z0 = center.z() - (width/2);
            z1 = center.z() + (width/2);
            k = center.x();

            float t = (k - r.Origin().x()) / r.Direction().x();

            if (t < t_min || t > t_max)
                return false;

            float y = r.Origin().y() + t*r.Direction().y();
            float z = r.Origin().z() + t*r.Direction().z();
            if (y < y0 || y > y1 || z < z0 || z > z1)
                return false;

            rec.u = (y-y0)/(y1-y0);
            rec.v = (z-z0)/(z1-z0);
            rec.t = t;
            Vec3 outward_normal = Vec3(1.0f, 0.0f, 0.0f);
            rec.SetFaceNormal(r, outward_normal);
            rec.mat_ptr = mat_ptr;
            rec.p = r.PointAtParameter(t);

            return true;
        }
    }

private:
    __device__ static inline void GetSphereUV(const Vec3& p, float& u, float& v)
    {
        float theta = acos(-p.y());
        float phi = atan2(-p.z(), p.x()) + PI;
        u = phi / (2 * PI);
        v = theta / PI;
    }
};

__forceinline__ __host__ const char *GetTextForEnum(int enumVal)
{
    switch(enumVal)
    {
    case Hitt::sphere:
        return "Sphere ";
    case Hitt::xy_rect:
        return "XY Rectangle ";
    case Hitt::xz_rect:
        return "XZ Rectangle ";
    case Hitt::yz_rect:
        return "YZ Rectangle ";

    default:
        return "Not recognized.. ";
    }
}