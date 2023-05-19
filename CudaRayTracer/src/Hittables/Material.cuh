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

    __device__ inline bool Scatter(const Ray& r, const HitRecord& rec, Vec3& attenuation, Ray& scattered,
                                   curandState* local_rand_state) const
    {
        Vec3 target = rec.p + rec.normal + RandomInUnitSphere(local_rand_state);
        scattered = Ray(rec.p, target - rec.p);
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        // switch (albedo->type)
        // {
        // case TextureType::CONSTANT:
        //     attenuation = albedo->Object->constant->value(rec.u, rec.v, rec.p);
        //     break;
        // case TextureType::CHECKER:
        //     attenuation = albedo->Object->checker->value(rec.u, rec.v, rec.p);
        //     break;
        // case TextureType::IMAGE:
        //     attenuation = albedo->Object->image->value(rec.u, rec.v, rec.p);
        //     break;
        // default:
        //     break;
        // }
        return true;
    }
};

class Metal
{
public:
    Texture* albedo;
    float fuzz;

    __host__ Metal(Texture* a, float f) : albedo(a), fuzz(f < 1 ? f : 1)
    {
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

    __device__ inline bool Scatter(const Ray& r, const HitRecord& rec, Vec3& attenuation, Ray& scattered,
                                   curandState* local_rand_state) const
    {
        Vec3 reflected = Reflect(UnitVector(r.Direction()), rec.normal);
        scattered = Ray(rec.p, reflected + fuzz * RandomInUnitSphere(local_rand_state));
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        // switch (albedo->type)
        // {
        // case TextureType::CONSTANT:
        //     attenuation = albedo->Object->constant->value(rec.u, rec.v, rec.p);
        //     break;
        // case TextureType::CHECKER:
        //     attenuation = albedo->Object->checker->value(rec.u, rec.v, rec.p);
        //     break;
        // case TextureType::IMAGE:
        //     attenuation = albedo->Object->image->value(rec.u, rec.v, rec.p);
        //     break;
        // default:
        //     break;
        // }
        return (Dot(scattered.Direction(), rec.normal) > 0);
    }
};

class Dielectric
{
public:
    Texture* albedo;
    float ir;

    __host__ Dielectric(float index_of_refraction) : ir(index_of_refraction)
    {
        if (albedo->texture == Tex::constant_texture) {
            albedo->even->color = Vec3(1.0f, 1.0f, 1.0f);
            albedo->odd->color = Vec3(0.0f, 0.0f, 0.0f);
        }
        albedo->color = Vec3(1.0f, 1.0f, 1.0f);
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
    int light_intensity;

    __host__ DiffuseLight(Texture* a, int l) : albedo(a), light_intensity(l)
    {
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

    __device__ inline bool Scatter(const Ray& r, const HitRecord& rec, Vec3& attenuation, Ray& scattered,
                                   curandState* local_rand_state) const
    {
        return false;
    }

    __device__ inline Vec3 Emitted(float u, float v, const Vec3& p) const
    {
        return light_intensity * albedo->value(u, v, p);
        // switch (albedo->type)
        // {
        // case TextureType::CONSTANT:
        //     return light_intensity * albedo->Object->constant->value(u, v, p);
        // case TextureType::CHECKER:
        //     return light_intensity * albedo->Object->checker->value(u, v, p);
        // case TextureType::IMAGE:
        //     return light_intensity * albedo->Object->image->value(u, v, p);
        // default:
        //     break;
        // }
    }
};

// enum Mat
// {
//     lambertian = 0,
//     metal,
//     dielectric,
//     diffuse_light,
// };

// class Material
// {
// public:
//     Texture* albedo;
//     float fuzz;
//     float ir;
//     int light_intensity;
//     Mat material;

// public:
//     __host__ Material()
//     {
//     }
//     __host__ Material(Texture* a, Mat m) : albedo(a), material(m)
//     {
//         if (albedo->texture == Tex::constant_texture) {
//             albedo->even->color = Vec3(1.0f, 1.0f, 1.0f);
//             albedo->odd->color = Vec3(0.0f, 0.0f, 0.0f);
//         }
//         else if (albedo->texture == Tex::checker_texture) {
//             albedo->color = Vec3(1.0f, 1.0f, 1.0f);
//         }
//         else if (albedo->texture == Tex::image_texture) {
//             albedo->even->color = Vec3(1.0f, 1.0f, 1.0f);
//             albedo->odd->color = Vec3(0.0f, 0.0f, 0.0f);
//             albedo->color = Vec3(1.0f, 1.0f, 1.0f);
//         }
//     }
//     __host__ Material(Texture* a, float f, Mat m) : albedo(a), fuzz(f < 1 ? f : 1), material(m)
//     {
//         if (albedo->texture == Tex::constant_texture) {
//             albedo->even->color = Vec3(1.0f, 1.0f, 1.0f);
//             albedo->odd->color = Vec3(0.0f, 0.0f, 0.0f);
//         }
//         else if (albedo->texture == Tex::checker_texture) {
//             albedo->color = Vec3(1.0f, 1.0f, 1.0f);
//         }
//         else if (albedo->texture == Tex::image_texture) {
//             albedo->even->color = Vec3(1.0f, 1.0f, 1.0f);
//             albedo->odd->color = Vec3(0.0f, 0.0f, 0.0f);
//             albedo->color = Vec3(1.0f, 1.0f, 1.0f);
//         }
//     }
//     __host__ Material(float index_of_refraction, Mat m) : ir(index_of_refraction), material(m)
//     {
//         if (albedo->texture == Tex::constant_texture) {
//             albedo->even->color = Vec3(1.0f, 1.0f, 1.0f);
//             albedo->odd->color = Vec3(0.0f, 0.0f, 0.0f);
//         }
//         albedo->color = Vec3(1.0f, 1.0f, 1.0f);
//     }
//     __host__ Material(Texture* a, int l, Mat m) : albedo(a), light_intensity(l), material(m)
//     {
//         if (albedo->texture == Tex::constant_texture) {
//             albedo->even->color = Vec3(1.0f, 1.0f, 1.0f);
//             albedo->odd->color = Vec3(0.0f, 0.0f, 0.0f);
//         }
//         else if (albedo->texture == Tex::checker_texture) {
//             albedo->color = Vec3(1.0f, 1.0f, 1.0f);
//         }
//         else if (albedo->texture == Tex::image_texture) {
//             albedo->even->color = Vec3(1.0f, 1.0f, 1.0f);
//             albedo->odd->color = Vec3(0.0f, 0.0f, 0.0f);
//             albedo->color = Vec3(1.0f, 1.0f, 1.0f);
//         }
//     }

//     __device__ inline bool Scatter(const Ray& r, const HitRecord& rec, Vec3& attenuation, Ray& scattered,
//                                    curandState* local_rand_state) const
//     {
//         if (material == Mat::lambertian) {
//             Vec3 target = rec.p + rec.normal + RandomInUnitSphere(local_rand_state);
//             scattered = Ray(rec.p, target - rec.p);
//             attenuation = albedo->value(rec.u, rec.v, rec.p);
//             return true;
//         }
//         else if (material == Mat::metal) {
//             Vec3 reflected = Reflect(UnitVector(r.Direction()), rec.normal);
//             scattered = Ray(rec.p, reflected + fuzz * RandomInUnitSphere(local_rand_state));
//             attenuation = albedo->value(rec.u, rec.v, rec.p);
//             return (Dot(scattered.Direction(), rec.normal) > 0);
//         }
//         else if (material == Mat::dielectric) {
//             Vec3 outward_normal;
//             Vec3 reflected = Reflect(r.Direction(), rec.normal);
//             float ni_over_nt;
//             attenuation = Vec3(1.0f, 1.0f, 1.0f);
//             Vec3 refracted;
//             float reflect_prob;
//             float cosine;
//             if (Dot(r.Direction(), rec.normal) > 0.0f) {
//                 outward_normal = -rec.normal;
//                 ni_over_nt = ir;
//                 cosine = Dot(r.Direction(), rec.normal) / r.Direction().Length();
//                 cosine = sqrt(1.0f - ir * ir * (1 - cosine * cosine));
//             }
//             else {
//                 outward_normal = rec.normal;
//                 ni_over_nt = 1.0f / ir;
//                 cosine = -Dot(r.Direction(), rec.normal) / r.Direction().Length();
//             }
//             if (Refract(r.Direction(), outward_normal, ni_over_nt, refracted))
//                 reflect_prob = Reflectance(cosine, ir);
//             else
//                 reflect_prob = 1.0f;
//             if (curand_uniform(local_rand_state) < reflect_prob)
//                 scattered = Ray(rec.p, reflected);
//             else
//                 scattered = Ray(rec.p, refracted);
//             return true;
//         }
//         else if (material == Mat::diffuse_light) {
//             return false;
//         }

//         return false;
//     }

//     __device__ inline Vec3 Emitted(float u, float v, const Vec3& p) const
//     {
//         return light_intensity * albedo->value(u, v, p);
//     }

// private:
//     __device__ static inline float Reflectance(float cosine, float ir)
//     {
//         // Use Schlick's approximation for reflectance.
//         auto r0 = (1.0f - ir) / (1.0f + ir);
//         r0 = r0 * r0;
//         return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
//     }
// };
