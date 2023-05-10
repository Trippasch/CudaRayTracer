#pragma once

typedef struct InputStruct
{
    float origin_x = 0.0f;
    float origin_y = 0.0f;
    float origin_z = 0.0f;

    float orientation_x = 0.0f;
    float orientation_y = 0.0f;
    float orientation_z = 0.0f;

    float up_x = 0.0f;
    float up_y = 0.0f;
    float up_z = 0.0f;

    float far_plane = 0.0f;
    float near_plane = 0.0f;
    float fov = 0.0f;

} InputStruct;
