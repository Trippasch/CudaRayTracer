#include "Math.cuh"

// clamp x to range [a, b]
__device__ float Clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

__device__ int Clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}