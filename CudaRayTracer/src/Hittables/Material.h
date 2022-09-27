#pragma once

#include "Hittable.h"

struct HitRecord;

class Material
{
public:
    __device__ virtual bool Scatter(const Ray& r, const HitRecord& rec,
        Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const = 0;
};

class Lambertian : public Material
{
public:
    Vec3 albedo;
public:
    __device__ Lambertian(const Vec3& a) : albedo(a) {}

    __device__ bool Scatter(const Ray& r, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const override
    {
        Vec3 target = rec.p + rec.normal + RandomInUnitSphere(local_rand_state);
        scattered = Ray(rec.p, target - rec.p);
        attenuation = albedo;

        return true;
    }
};

class Metal : public Material
{
public:
    Vec3 albedo;
    float fuzz;

public:
    __device__ Metal(const Vec3& a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    __device__ bool Scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const override
    {
        Vec3 reflected = Reflect(UnitVector(r_in.Direction()), rec.normal);
        scattered = Ray(rec.p, reflected + fuzz*RandomInUnitSphere(local_rand_state));
        attenuation = albedo;
        return (Dot(scattered.Direction(), rec.normal) > 0);
    }
};

class Dielectric : public Material
{
public:
    float ir; // Index of Refraction

public:
    __device__ Dielectric(float index_of_refraction) : ir(index_of_refraction) {}

    __device__ bool Scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const override
    {
        Vec3 outward_normal;
        Vec3 reflected = Reflect(r_in.Direction(), rec.normal);
        float ni_over_nt;
        attenuation = Vec3(1.0f, 1.0f, 1.0f);
        Vec3 refracted;
        float reflect_prob;
        float cosine;
        if (Dot(r_in.Direction(), rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ir;
            cosine = Dot(r_in.Direction(), rec.normal) / r_in.Direction().Length();
            cosine = sqrt(1.0f - ir*ir*(1-cosine*cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ir;
            cosine = -Dot(r_in.Direction(), rec.normal) / r_in.Direction().Length();
        }
        if (Refract(r_in.Direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = Reflectance(cosine, ir);
        else
            reflect_prob = 1.0f;
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = Ray(rec.p, reflected);
        else
            scattered = Ray(rec.p, refracted);
        return true;
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
