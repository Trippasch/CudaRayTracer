#pragma once

#include "../Utils/Vec3.h"

enum Tex {
    constant_texture = 0,
    checker_texture,
};

class Texture
{
public:
    Vec3 color;
    Texture* odd;
    Texture* even;
    Tex texture;

public:
    __host__ Texture() {}
    __host__ Texture(Vec3 c, Tex t) : color(c), texture(t) {}
    __host__ Texture(Texture* t0, Texture* t1, Tex t) : even(t0), odd(t1), texture(t) {}

    __device__ inline Vec3 value(float u, float v, const Vec3& p) const
    {
        if (texture == 0) {
            return color;
        }
        else if (texture == 1) {
            float sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
            if (sines < 0) {
                return odd->value(u, v, p);
            }
            else {
                return even->value(u, v, p);
            }
        }
    }
};