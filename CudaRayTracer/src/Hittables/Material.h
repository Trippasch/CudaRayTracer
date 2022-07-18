#pragma once

#include "Hittable.h"

struct hit_record;

class material
{
public:
    __device__ virtual bool scatter(const ray& r, const hit_record& rec,
        float3& attenuation, ray& scattered, curandState* local_rand_state) const = 0;
};

class lambertian : public material
{
public:
    float3 albedo;
public:
    __device__ lambertian(const float3& a) : albedo(a) {}

    __device__ bool scatter(const ray& r, const hit_record& rec, float3& attenuation, ray& scattered, curandState* local_rand_state) const override
    {
        float3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
        scattered = ray(rec.p, target - rec.p);
        attenuation = albedo;

        return true;
    }
};

class metal : public material
{
public:
    float3 albedo;
    float fuzz;

public:
    __device__ metal(const float3& a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, float3& attenuation, ray& scattered, curandState* local_rand_state) const override
    {
        float3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }
};

class dielectric : public material
{
public:
    float ir; // Index of Refraction

public:
    __device__ dielectric(float index_of_refraction) : ir(index_of_refraction) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, float3& attenuation, ray& scattered, curandState* local_rand_state) const override
    {
        float3 outward_normal;
        float3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = make_float3(1.0, 1.0, 1.0);
        float3 refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ir;
            cosine = dot(r_in.direction(), rec.normal) / length(r_in.direction());
            cosine = sqrt(1.0f - ir*ir*(1-cosine*cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ir;
            cosine = -dot(r_in.direction(), rec.normal) / length(r_in.direction());
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = reflectance(cosine, ir);
        else
            reflect_prob = 1.0f;
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = ray(rec.p, reflected);
        else
            scattered = ray(rec.p, refracted);
        return true;
    }

private:
    __device__ static float reflectance(float cosine, float ir)
    {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1.0f - ir) / (1.0f + ir);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
    }
};
