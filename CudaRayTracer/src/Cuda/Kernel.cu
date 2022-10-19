#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <algorithm>

#include "../Hittables/HittableList.h"
#include "../Hittables/Sphere.h"
#include "../Hittables/Material.h"
#include "../Utils/SharedStructs.h"

// clamp x to range [a, b]
__device__ inline float Clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

__device__ inline int Clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}

// convert floating point rgb color to 8-bit integer
__device__ inline int RgbToInt(float r, float g, float b)
{
    r = Clamp(r, 0.0f, 255.0f);
    g = Clamp(g, 0.0f, 255.0f);
    b = Clamp(b, 0.0f, 255.0f);
    float a = 255.0f;
    return (int(a) << 24) | (int(b) << 16) | (int(g) << 8) | int(r);
}

// the reverse
__device__ inline Vec3 IntToRgb(int val)
{
    float r = val % 256;
    float g = (val % (256 * 256)) / 256;
    float b = val / (256 * 256);
    return Vec3(r, g, b);
}

__device__ inline bool HitSphere(const Vec3& center, float radius, const Ray& r)
{
    Vec3 oc = r.Origin() - center;
    float a = Dot(r.Direction(), r.Direction());
    float b = 2.0f * Dot(oc, r.Direction());
    float c = Dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0f * a * c;

    return (discriminant > 0.0f);
}

__device__ inline Vec3 color(const Ray& r, Hittable** world, Hittable *head, const int max_depth, curandState* local_rand_state) {
    Ray cur_ray = r;
    Vec3 cur_attenuation = Vec3(1.0f, 1.0f, 1.0f);
    for (int i = 0; i < max_depth; i++) {
        HitRecord rec;
        if ((*world)->Hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            Ray scattered;
            Vec3 attenuation;
            if (rec.mat_ptr->Scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation = attenuation * cur_attenuation;
                cur_ray = scattered;
            }
            else {
                return Vec3(0.0f, 0.0f, 0.0f);
            }
        }
        else {
            Vec3 unit_direction = UnitVector(cur_ray.Direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            Vec3 c = (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
            return cur_attenuation * c;
        }
    }
    return Vec3(0.0f, 0.0f, 0.0f); // exceeded recursion
}

__device__ inline void GetXYZCoords(int& x, int& y, int& z)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int bt = blockDim.z;
    x = blockIdx.x * bw + tx;
    y = blockIdx.y * bh + ty;
    z = blockIdx.z + bt * tz;
}

__global__ void Kernel(unsigned int* pos, unsigned int width, unsigned int height, const unsigned int samples_per_pixel, const unsigned int max_depth, Hittable** world, Hittable* head, curandState* rand_state, InputStruct inputs)
{
    extern __shared__ uchar4 sdata[];

    int x, y, z;
    GetXYZCoords(x, y, z);

    if ((x >= width) || (y >= height))
        return;

    unsigned int pixel_index = (y * width + x);

    curandState local_rand_state = rand_state[pixel_index];

    Vec3 col = Vec3(0.0f, 0.0f, 0.0f);

    Vec3 origin = Vec3(inputs.origin_x, inputs.origin_y, inputs.origin_z);
    Vec3 forwardV = Vec3(inputs.orientation_x, inputs.orientation_y, inputs.orientation_z);
    Vec3 upV = Vec3(inputs.up_x, inputs.up_y, inputs.up_z);
    Vec3 rightV = Normalize(Cross(upV, forwardV));

    float distFirstPlane = inputs.fov * 0.1f;

    Vec3 center = Vec3(width / 2.0f, height / 2.0f, 0.0f);
    Vec3 distFromCenter = ((x - center.x()) / width) * rightV + ((center.y() - y) / width) * upV;
    Vec3 startPos = (inputs.near * distFromCenter) + origin + (distFirstPlane * forwardV);
    Vec3 secondPlanePos = (inputs.far * distFromCenter) + (inputs.fov * forwardV) + origin;

    Vec3 dirVector = Normalize(secondPlanePos - startPos);

    Ray r = Ray(startPos, dirVector);
    // col = col + color(r, world, max_depth, &local_rand_state);

    for (int s = 0; s < samples_per_pixel; s++)
    {
        // calculate uv coordinates
    //    float u = (float)(x + curand_uniform(&local_rand_state)) / (float)(width);
    //    float v = (float)(y + curand_uniform(&local_rand_state)) / (float)(height);
       col = col + color(r, world, head, max_depth, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;

    col = col / (float)(samples_per_pixel);
    col.e[0] = 255.0f * sqrt(col.x());
    col.e[1] = 255.0f * sqrt(col.y());
    col.e[2] = 255.0f * sqrt(col.z());

    // write output vertex
    pos[pixel_index] = RgbToInt(col.x(), col.y(), col.z());
}

__global__ void RandInit(curandState* rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        curand_init(1984, 0, 0, rand_state);
}

__global__ void RenderInit(unsigned int window_width, unsigned int window_height, curandState* rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= window_width) || (j >= window_height))
        return;

    unsigned int pixel_index = (j * window_width + i);
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void CreateWorld(Hittable** d_list, Hittable** d_world, const float aspect_ratio, curandState* rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;

        int i = 0;

        //d_list[i++] = new Sphere(Vec3(0, -1000.0, -1), 1000,
        //    new Lambertian(Vec3(0.5, 0.5, 0.5)));
        //for (int a = -11; a < 11; a++) {
        //    for (int b = -11; b < 11; b++) {
        //        float choose_mat = RND;
        //        Vec3 center = Vec3(a + RND, 0.2, b + RND);
        //        if (choose_mat < 0.8f) {
        //            d_list[i++] = new Sphere(center, 0.2,
        //                new Lambertian(Vec3(RND * RND, RND * RND, RND * RND)));
        //        }
        //        else if (choose_mat < 0.95f) {
        //            d_list[i++] = new Sphere(center, 0.2,
        //                new Metal(Vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
        //        }
        //        else {
        //            d_list[i++] = new Sphere(center, 0.2, new Dielectric(1.5));
        //        }
        //    }
        //}
        //d_list[i++] = new Sphere(Vec3(0, 1, 0), 1.0, new Dielectric(1.5));
        //d_list[i++] = new Sphere(Vec3(0, 1, 0), -0.9, new Dielectric(1.5));
        //d_list[i++] = new Sphere(Vec3(-4, 1, 0), 1.0, new Lambertian(Vec3(0.4, 0.2, 0.1)));
        //d_list[i++] = new Sphere(Vec3(4, 1, 0), 1.0, new Metal(Vec3(0.7, 0.6, 0.5), 0.0));

        d_list[i++] = new Sphere(Vec3(0, -100.5, 0), 100, new Lambertian(Vec3(0.8, 0.8, 0.0)));
        d_list[i++] = new Sphere(Vec3(0, 0, -1), 0.5, new Lambertian(Vec3(0.1, 0.2, 0.5)));
        d_list[i++] = new Sphere(Vec3(1, 0, -1), 0.5, new Metal(Vec3(0.8, 0.6, 0.2), 0.0));
        d_list[i++] = new Sphere(Vec3(-1, 0, -1), 0.5, new Dielectric(1.5f));
        d_list[i++] = new Sphere(Vec3(-1, 0, -1), -0.45, new Dielectric(1.5f));

        *rand_state = local_rand_state;
        // *d_world = new HittableList(d_list, i);
    }
}

__global__ void FreeWorld(Hittable** d_list, Hittable** d_world, const unsigned int num_hittables) {
    for (int i = 0; i < num_hittables; i++) {
        delete ((Sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* d_world;
}

extern "C"
void LaunchKernel(dim3 grid, dim3 block, int sbytes, unsigned int* pos, unsigned int window_width, unsigned int window_height, const unsigned int samples_per_pixel, const unsigned int max_depth, Hittable** d_world, Hittable* d_head, curandState* d_rand_state, InputStruct inputs)
{
    Kernel << < grid, block, sbytes >> > (pos, window_width, window_height, samples_per_pixel, max_depth, d_world, d_head, d_rand_state, inputs);
    cudaDeviceSynchronize();
}

extern "C"
void LaunchRandInit(curandState* d_rand_state2)
{
    RandInit << < 1, 1 >> > (d_rand_state2);
    cudaDeviceSynchronize();
}

extern "C"
void LaunchRenderInit(dim3 grid, dim3 block, unsigned int window_width, unsigned int window_height, curandState* d_rand_state)
{
    RenderInit << < grid, block >> > (window_width, window_height, d_rand_state);
    cudaDeviceSynchronize();
}

extern "C"
void LaunchCreateWorld(Hittable** d_list, Hittable** d_world, const float aspect_ratio, curandState* d_rand_state2)
{
    CreateWorld << < 1, 1 >> > (d_list, d_world, aspect_ratio, d_rand_state2);
    cudaDeviceSynchronize();
}

extern "C"
void LaunchFreeWorld(Hittable** d_list, Hittable** d_world, const unsigned int num_hittables)
{
    FreeWorld << < 1, 1 >> > (d_list, d_world, num_hittables);
    cudaDeviceSynchronize();
}