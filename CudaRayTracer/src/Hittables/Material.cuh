#pragma once

#include "Hittable.cuh"
#include "Texture.cuh"

enum Mat {
    lambertian = 0,
    metal,
    dielectric,
    diffuse_light,
};

class Material
{
public:
    Texture* albedo;
    float fuzz;
    float ir;
    int light_intensity;
    Mat material;
    Texture* emit;

public:
    __host__ Material() {}
    __host__ Material(Texture* a, Mat m) : albedo(a), material(m) {
        if (albedo->texture == Tex::constant_texture) {
            albedo->even->color = Vec3(1.0f, 1.0f, 1.0f);
            albedo->odd->color = Vec3(0.0f, 0.0f, 0.0f);
        }
        else if (albedo->texture == Tex::checker_texture) {
            albedo->color = Vec3(1.0f, 1.0f, 1.0f);
        }
        else if (albedo->texture == Tex::image_texture) {
            albedo->even->color = Vec3(1.0f, 1.0f, 1.0f);
            albedo->odd->color = Vec3(0.0f, 0.0f, 0.0f);
            albedo->color = Vec3(1.0f, 1.0f, 1.0f);
        }
    }
    __host__ Material(Texture* a, float f, Mat m) : albedo(a), fuzz(f < 1 ? f : 1), material(m) {
        if (albedo->texture == Tex::constant_texture) {
            albedo->even->color = Vec3(1.0f, 1.0f, 1.0f);
            albedo->odd->color = Vec3(0.0f, 0.0f, 0.0f);
        }
        else if (albedo->texture == Tex::checker_texture) {
            albedo->color = Vec3(1.0f, 1.0f, 1.0f);
        }
        else if (albedo->texture == Tex::image_texture) {
            albedo->even->color = Vec3(1.0f, 1.0f, 1.0f);
            albedo->odd->color = Vec3(0.0f, 0.0f, 0.0f);
            albedo->color = Vec3(1.0f, 1.0f, 1.0f);
        }
    }
    __host__ Material(float index_of_refraction , Mat m) : ir(index_of_refraction), material(m) {
        if (albedo->texture == Tex::constant_texture) {
            albedo->even->color = Vec3(1.0f, 1.0f, 1.0f);
            albedo->odd->color = Vec3(0.0f, 0.0f, 0.0f);
        }
        albedo->color = Vec3(1.0f, 1.0f, 1.0f);
    }
    __host__ Material(Texture* e, int i, Mat m) : emit(e), light_intensity(i), material(m) {}

    __device__ inline bool Scatter(const Ray& r, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const
    {
        if (material == 0) {
            Vec3 target = rec.p + rec.normal + RandomInUnitSphere(local_rand_state);
            scattered = Ray(rec.p, target - rec.p);
            attenuation = albedo->value(rec.u, rec.v, rec.p);
            return true;
        }
        else if (material == 1) {
            Vec3 reflected = Reflect(UnitVector(r.Direction()), rec.normal);
            scattered = Ray(rec.p, reflected + fuzz*RandomInUnitSphere(local_rand_state));
            attenuation = albedo->value(rec.u, rec.v, rec.p);
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
        else if (material == 3) {
            return false;
        }
    }

    __device__ inline Vec3 Emitted(float u, float v, const Vec3& p) const
    {
        return light_intensity * emit->value(u, v, p);
    }

private:
    __device__ static inline float Reflectance(float cosine, float ir)
    {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1.0f - ir) / (1.0f + ir);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
    }
};

__forceinline__ __host__ const char *GetTextForEnum(int enumVal)
{
    switch(enumVal)
    {
    case Mat::lambertian:
        return "Lambertian";
    case Mat::metal:
        return "Metal";
    case Mat::dielectric:
        return "Dielectric";
    case Mat::diffuse_light:
        return "Diffuse Light";

    default:
        return "Not recognized..";
    }
}