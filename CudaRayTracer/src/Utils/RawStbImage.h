#pragma once

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

inline unsigned char* LoadImage(const char* filename, unsigned char* data, int* width, int* height, int* nr)
{
    data = stbi_load(filename, width, height, nr, 0);

    if (!data) {
        RT_ERROR("ERROR: Could not load texture image file {0}", filename);
        width = height = 0;
    }

    return data;
}