#pragma once

#include "Hittables/Hittable.cuh"
#include "Hittables/Texture.cuh"

typedef enum MaterialType
{
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    DIFFUSELIGHT
} MaterialType;

class Lambertian;
class Metal;
class Dielectric;
class DiffuseLight;

class Material
{
public:
    MaterialType type;

    union ObjectUnion {
        Lambertian* lambertian;
        Metal* metal;
        Dielectric* dielectric;
        DiffuseLight* diffuse_light;
    };

    ObjectUnion* Object;
};

class Lambertian
{
public:
    Texture* albedo;

    __host__ Lambertian(Texture* a) : albedo(a)
    {
    }

    __device__ inline bool Scatter(const Ray& r, const HitRecord& rec, Vec3& attenuation, Ray& scattered,
                                   curandState* local_rand_state) const
    {
        Vec3 target = rec.p + rec.normal + RandomInUnitSphere(local_rand_state);
        scattered = Ray(rec.p, target - rec.p);
        switch (albedo->type) {
        case TextureType::CONSTANT:
            attenuation = albedo->Object->constant->value(rec.u, rec.v, rec.p);
            break;
        case TextureType::CHECKER:
            attenuation = albedo->Object->checker->value(rec.u, rec.v, rec.p);
            break;
        case TextureType::IMAGE:
            attenuation = albedo->Object->image->value(rec.u, rec.v, rec.p);
            break;
        default:
            break;
        }
        return true;
    }
};

class Metal
{
public:
    Texture* albedo;
    float fuzz = 0.0f;

    __host__ Metal(Texture* a, float f) : albedo(a), fuzz(f < 1 ? f : 1)
    {
    }

    __device__ inline bool Scatter(const Ray& r, const HitRecord& rec, Vec3& attenuation, Ray& scattered,
                                   curandState* local_rand_state) const
    {
        Vec3 reflected = Reflect(UnitVector(r.Direction()), rec.normal);
        scattered = Ray(rec.p, reflected + fuzz * RandomInUnitSphere(local_rand_state));
        switch (albedo->type) {
        case TextureType::CONSTANT:
            attenuation = albedo->Object->constant->value(rec.u, rec.v, rec.p);
            break;
        case TextureType::CHECKER:
            attenuation = albedo->Object->checker->value(rec.u, rec.v, rec.p);
            break;
        case TextureType::IMAGE:
            attenuation = albedo->Object->image->value(rec.u, rec.v, rec.p);
            break;
        default:
            break;
        }
        return (Dot(scattered.Direction(), rec.normal) > 0);
    }
};

class Dielectric
{
public:
    float ir = 0.0f;

    __host__ Dielectric(float index_of_refraction) : ir(index_of_refraction)
    {
    }

    __device__ inline bool Scatter(const Ray& r, const HitRecord& rec, Vec3& attenuation, Ray& scattered,
                                   curandState* local_rand_state) const
    {
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
            cosine = sqrt(1.0f - ir * ir * (1 - cosine * cosine));
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

private:
    __device__ static inline float Reflectance(float cosine, float ir)
    {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1.0f - ir) / (1.0f + ir);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
    }
};

class DiffuseLight
{
public:
    Texture* albedo;
    int light_intensity = 3;

    __host__ DiffuseLight(Texture* a, int l) : albedo(a), light_intensity(l)
    {
    }

    __device__ inline bool Scatter(const Ray& r, const HitRecord& rec, Vec3& attenuation, Ray& scattered,
                                   curandState* local_rand_state) const
    {
        return false;
    }

    __device__ inline Vec3 Emitted(float u, float v, const Vec3& p) const
    {
        switch (albedo->type) {
        case TextureType::CONSTANT:
            return light_intensity * albedo->Object->constant->value(u, v, p);
        case TextureType::CHECKER:
            return light_intensity * albedo->Object->checker->value(u, v, p);
        case TextureType::IMAGE:
            return light_intensity * albedo->Object->image->value(u, v, p);
        default:
            break;
        }
    }
};