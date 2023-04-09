#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#include "../Hittables/HittableList.cuh"
#include "../Hittables/Sphere.cuh"
#include "../Hittables/Material.cuh"
#include "../Utils/SharedStructs.h"
#include "../Utils/Math.cu"

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

__device__ bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec, Sphere** world, unsigned int world_size)
{
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < world_size; i++) {
        if (world[i]->Hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

__device__ inline Vec3 color(const Ray& r, Sphere** world, unsigned int world_size, int max_depth, curandState* local_rand_state) {

    // HitRecord rec;

    // // If we've exceeded the ray bounce limit, no more light is gathered.
    // if (max_depth <= 0)
    //     return Vec3(0.0f, 0.0f, 0.0f);

    // // If the ray hits nothing, return the background color.
    // if (!Hit(r, 0.001f, FLT_MAX, rec, world, world_size))
    //     return Vec3(1.0f, 0.0f, 0.0f);

    // Ray scattered;
    // Vec3 attenuation;
    // Vec3 emitted = rec.mat_ptr->Emitted(rec.u, rec.v, rec.p);

    // if (!rec.mat_ptr->Scatter(r, rec, attenuation, scattered, local_rand_state))
    //     return emitted;

    // return emitted + attenuation * color(scattered, world, world_size, max_depth - 1, local_rand_state);

    ///////////////////////////////////////////////////////////////////////////

    // Ray cur_ray = r;
    // Vec3 cur_attenuation = Vec3(1.0f, 1.0f, 1.0f);

    // Vec3 background(0, 0, 0);
    // Vec3 emitted_stack[8];
    // Vec3 ray_color_stack[8];
    // Vec3 attenuation_stack[8];
    // int index = 0;

    // for (int i = 0; i < max_depth; i++) {
    //     HitRecord rec;
    //     if (!Hit(cur_ray, 0.001f, FLT_MAX, rec, world, world_size)) {
    //         ray_color_stack[i] = background;
    //         index = i;
    //         break;
    //     }

    //     Ray scattered;
    //     Vec3 attenuation;
    //     Vec3 emitted = rec.mat_ptr->Emitted(rec, 0, 0, Vec3(0, 0, 0));
    //     emitted_stack[i] = emitted;
    //     if (!(rec.mat_ptr)->Scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
    //         ray_color_stack[i] = emitted;
    //         index = i;
    //         break;
    //     }
    //     cur_ray = scattered;
    //     attenuation_stack[i] = attenuation;
    // }

    // for (int i = index-1; i >= 0; --i) {
    //     ray_color_stack[i] = emitted_stack[i] + attenuation_stack[i] * ray_color_stack[i + 1];
    // }
    // return ray_color_stack[0];

    ///////////////////////////////////////////////////////////////////////////

    Ray cur_ray = r;
    Vec3 cur_attenuation = Vec3(1.0f, 1.0f, 1.0f);

    for (int i = 0; i < max_depth; i++) {
        HitRecord rec;
        Vec3 emitted = Vec3(0.0f, 0.0f, 0.0f);
        if (Hit(cur_ray, 0.001f, FLT_MAX, rec, world, world_size)) {
            Ray scattered;
            Vec3 attenuation;

            if (rec.mat_ptr->material == Mat::diffuse_light) {
                emitted = (rec.mat_ptr)->Emitted(rec.u, rec.v, rec.p);
            }

            if ((rec.mat_ptr)->Scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation = attenuation * cur_attenuation;
                cur_ray = scattered;

                // return emitted + cur_attenuation;
            }
            else {
                return emitted;
                // return Vec3(0.0f, 0.0f, 0.0f);
            }
        }
        else {
            // return Vec3(0.0f, 0.0f, 0.0f);
            Vec3 unit_direction = UnitVector(cur_ray.Direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            Vec3 c = (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
            return emitted + cur_attenuation * c;
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

// #define THREADS_PER_BLOCK          256
// #if __CUDA_ARCH__ >= 200
//     #define MY_KERNEL_MAX_THREADS  (2 * THREADS_PER_BLOCK)
//     #define MY_KERNEL_MIN_BLOCKS   3
// #else
//     #define MY_KERNEL_MAX_THREADS  THREADS_PER_BLOCK
//     #define MY_KERNEL_MIN_BLOCKS   2
// #endif

__global__
// __launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
void Kernel(unsigned int* pos, unsigned int width, unsigned int height, const unsigned int samples_per_pixel, const unsigned int max_depth, Sphere** world, unsigned int world_size, curandState* rand_state, InputStruct inputs)
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

    Vec3 center = Vec3(width / 2.0f, height / 2.0f, 0.0f);

    for (int s = 0; s < samples_per_pixel; s++) {
        // calculate uv coordinates
        float u = (float)((x - center.x()) + curand_uniform(&local_rand_state)) / (float)(width);
        float v = (float)((center.y() - y) + curand_uniform(&local_rand_state)) / (float)(width);
        Vec3 distFromCenter = (u * rightV) + (v * upV);
        Vec3 startPos = (inputs.near_plane * distFromCenter) + origin + (inputs.fov * forwardV);
        Vec3 secondPlanePos = (inputs.far_plane * distFromCenter) + ((1.0f/inputs.fov * 10.0f) * forwardV) + origin;
        Vec3 dirVector = Normalize(secondPlanePos - startPos);

        Ray r = Ray(startPos, dirVector);
        col = col + color(r, world, world_size, max_depth, &local_rand_state);
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

// __global__ void CreateWorld(Sphere** d_list, Sphere** d_world, curandState* rand_state)
// {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         curandState local_rand_state = *rand_state;

//         int i = 0;

//         d_list[i++] = new Sphere(Vec3(0, -1000.0, -1), 1000,
//            new Material(Vec3(0.5, 0.5, 0.5), Mat::lambertian));
//         for (int a = -2; a < 2; a++) {
//            for (int b = -2; b < 2; b++) {
//                float choose_mat = RND;
//                Vec3 center = Vec3(a + RND, 0.2, b + RND);
//                if (choose_mat < 0.8f) {
//                    d_list[i++] = new Sphere(center, 0.2,
//                        new Material(Vec3(RND * RND, RND * RND, RND * RND), Mat::lambertian));
//                }
//                else if (choose_mat < 0.95f) {
//                    d_list[i++] = new Sphere(center, 0.2,
//                        new Material(Vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND, Mat::metal));
//                }
//                else {
//                    d_list[i++] = new Sphere(center, 0.2, new Material(1.5, Mat::dielectric));
//                }
//            }
//         }
//         d_list[i++] = new Sphere(Vec3(0, 1, 0), 1.0, new Material(1.5, Mat::dielectric));
//         d_list[i++] = new Sphere(Vec3(-4, 1, 0), 1.0, new Material(Vec3(0.4, 0.2, 0.1), Mat::lambertian));
//         d_list[i++] = new Sphere(Vec3(4, 1, 0), 1.0, new Material(Vec3(0.7, 0.6, 0.5), 0.0, Mat::metal));

//         // d_list[i++] = new Sphere(Vec3(0, -100.5, 0), 100, new Material(Vec3(0.8, 0.8, 0.0), Mat::lambertian));
//         // d_list[i++] = new Sphere(Vec3(0, 0, -1), 0.5, new Material(Vec3(0.1, 0.2, 0.5), Mat::lambertian));
//         // d_list[i++] = new Sphere(Vec3(1, 0, -1), 0.5, new Material(Vec3(0.8, 0.6, 0.2), 0.0, Mat::metal));
//         // d_list[i++] = new Sphere(Vec3(-1, 0, -1), 0.5, new Material(1.5f, Mat::dielectric));
//         // d_list[i++] = new Sphere(Vec3(-1, 0, -1), -0.45, new Material(1.5f, Mat::dielectric));

//         *rand_state = local_rand_state;
//         *d_world = new HittableList(d_list, i);
//     }
// }

// __global__ void FreeWorld(Sphere** d_list, Sphere** d_world, const unsigned int num_hittables) {
//     for (int i = 0; i < num_hittables; i++) {
//         delete ((Sphere*)d_list[i])->mat_ptr;
//         delete d_list[i];
//     }
//     delete* d_world;
// }

extern "C"
void LaunchKernel(unsigned int* pos, unsigned int image_width, unsigned int image_height, const unsigned int samples_per_pixel, const unsigned int max_depth, HittableList* world, curandState* d_rand_state, InputStruct inputs)
{
    // Calculate grid size
    dim3 block(16, 16, 1);
    dim3 grid(image_width / block.x, image_height / block.y, 1);
    size_t sbytes = 0;

    Sphere** d_world;
    cudaMallocManaged((void**)&d_world, world->objects.size() * sizeof(Sphere*));

    for (int i = 0; i < world->objects.size(); i++) {
        d_world[i] = world->objects[i];
    }

    Kernel << < grid, block, sbytes >> > (pos, image_width, image_height, samples_per_pixel, max_depth, d_world, world->objects.size(), d_rand_state, inputs);
    cudaDeviceSynchronize();

    cudaFree(d_world);
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

// extern "C"
// void LaunchCreateWorld(Sphere** d_list, Sphere** d_world, curandState* d_rand_state2)
// {
//     CreateWorld << < 1, 1 >> > (d_list, d_world, d_rand_state2);
//     cudaDeviceSynchronize();
// }

// extern "C"
// void LaunchFreeWorld(Sphere** d_list, Sphere** d_world, const unsigned int num_hittables)
// {
//     FreeWorld << < 1, 1 >> > (d_list, d_world, num_hittables);
//     cudaDeviceSynchronize();
// }