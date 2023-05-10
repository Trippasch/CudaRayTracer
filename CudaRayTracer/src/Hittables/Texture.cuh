#pragma once

#include "Core/Log.h"

#include "Utils/Math.cuh"

enum Tex {
    constant_texture = 0,
    checker_texture,
    image_texture,
};

class Texture
{
public:
    Vec3 color;
    Texture* odd;
    Texture* even;
    Tex texture;
    unsigned char *data = nullptr;
    const char *path = nullptr;
    int width, height;
    const static int bytes_per_pixel = 3;

public:
    __host__ Texture() : color(Vec3(1.0f, 1.0f, 1.0f)), texture(Tex::constant_texture) {}
    __host__ Texture(Vec3 c, Tex t) : color(c), texture(t) {}
    __host__ Texture(Texture* t0, Texture* t1, Tex t) : even(t0), odd(t1), texture(t) {}
    __host__ Texture(unsigned char* d, int w, int h, Tex t) : data(d), width(w), height(h), texture(t)
    {
        bytes_per_scanline = bytes_per_pixel * width;
    }

    __device__ inline Vec3 value(float u, float v, const Vec3& p) const
    {
        if (texture == Tex::constant_texture) {
            return color;
        }
        else if (texture == Tex::checker_texture) {
            float sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
            if (sines < 0) {
                return odd->value(u, v, p);
            }
            else {
                return even->value(u, v, p);
            }
        }
        else if (texture == Tex::image_texture) {
            if (data == nullptr)
                return Vec3(0, 1, 1);

            // Clamp input texture coordinates to [0,1] x [1,0]
            u = Clamp(u, 0.0f, 1.0f);
            v = 1.0f - Clamp(v, 0.0f, 1.0f); // Flip V to image coordinates

            int i = static_cast<int>(u * width);
            int j = static_cast<int>(v * height);

            // Clamp integer mapping, since actual coordinates should be less than 1.0
            if (i >= width)
                i = width - 1;
            if (j >= height)
                j = height - 1;

            const float color_scale = 1.0f / 255.0f;
            unsigned char* pixel = data + j * bytes_per_scanline + i * bytes_per_pixel;

            return Vec3(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
        }
    }

private:
    int bytes_per_scanline;
};