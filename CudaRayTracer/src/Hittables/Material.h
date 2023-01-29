#pragma once

#include "Hittable.h"

enum Mat {
    lambertian = 0,
    metal,
    dielectric
};

class Material
{
public:
    Vec3 albedo;
    float fuzz;
    float ir;
    Mat material;
public:
    __host__ Material(const Vec3& a, Mat m) : albedo(a), material(m) {}

    __host__ Material(const Vec3& a, float f, Mat m) : albedo(a), fuzz(f), material(m) {}
    __host__ Material(float index_of_refraction , Mat m) : ir(index_of_refraction), material(m) {}

    __device__ bool Scatter(const Ray& r, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const
    {
        if (material == 0) {
            Vec3 target = rec.p + rec.normal + RandomInUnitSphere(local_rand_state);
            scattered = Ray(rec.p, target - rec.p);
            attenuation = albedo;
            return true;
        }
        else if (material == 1) {
            Vec3 reflected = Reflect(UnitVector(r.Direction()), rec.normal);
            scattered = Ray(rec.p, reflected + fuzz*RandomInUnitSphere(local_rand_state));
            attenuation = albedo;
            return (Dot(scattered.Direction(), rec.normal) > 0);
        }
        else if (material == 2) {
            Vec3 outward_normal;
            Vec3 reflected = Reflect(r.Direction(), rec.normal);
            float ni_over_nt;
            attenuation = Vec3(1.0f, 1.0f, 1.0f);
            Vec3 refracted;
            float reflect_prob;
            float cosine;
            if (Dot(r.Direction(), rec.normal) > 0.0f) {
                outward_normal = -rec.normal;
                ni_over_nt = ir;
                cosine = Dot(r.Direction(), rec.normal) / r.Direction().Length();
                cosine = sqrt(1.0f - ir*ir*(1-cosine*cosine));
            }
            else {
                outward_normal = rec.normal;
                ni_over_nt = 1.0f / ir;
                cosine = -Dot(r.Direction(), rec.normal) / r.Direction().Length();
            }
            if (Refract(r.Direction(), outward_normal, ni_over_nt, refracted))
                reflect_prob = Reflectance(cosine, ir);
            else
                reflect_prob = 1.0f;
            if (curand_uniform(local_rand_state) < reflect_prob)
                scattered = Ray(rec.p, reflected);
            else
                scattered = Ray(rec.p, refracted);
            return true;
        }
    }

private:
    __device__ static float Reflectance(float cosine, float ir)
    {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1.0f - ir) / (1.0f + ir);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
    }
};