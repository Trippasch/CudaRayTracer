#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include "Hittables/Hittable.cuh"
#include "Hittables/HittableList.cuh"
#include "Hittables/Material.cuh"
#include "Utils/SharedStructs.h"

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

// __device__ bool Hit(const Ray& r, float t_min, float t_max, HitRecord& rec, Hittable** world, unsigned int
// world_size)
// {
//     HitRecord temp_rec;
//     bool hit_anything = false;
//     float closest_so_far = t_max;

//     for (int i = 0; i < world_size; i++) {
//         if (world[i]->Hit(r, t_min, closest_so_far, temp_rec)) {
//             hit_anything = true;
//             closest_so_far = temp_rec.t;
//             rec = temp_rec;
//         }
//     }

//     return hit_anything;
// }

__device__ inline Vec3 color(const Ray& r, Hittable* world, int max_depth, curandState* local_rand_state)
{
    Ray cur_ray = r;
    Vec3 cur_attenuation = Vec3(1.0f, 1.0f, 1.0f);
    Vec3 background = Vec3(0.0f, 0.0f, 0.0f);

    HitRecord rec;

    for (int i = 0; i < max_depth; i++) {
        if (!world->Object->bvh_node->Hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            Vec3 unit_direction = UnitVector(cur_ray.Direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            Vec3 c = (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
            return cur_attenuation * c;
        }
        else {
            Vec3 emitted = Vec3(0.0f, 0.0f, 0.0f);
            Ray scattered;
            Vec3 attenuation;

            switch (rec.mat_ptr->type) {
            case MaterialType::LAMBERTIAN:
                if (!rec.mat_ptr->Object->lambertian->Scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                    return emitted * cur_attenuation;
                }
                break;
            case MaterialType::METAL:
                if (!rec.mat_ptr->Object->metal->Scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                    return emitted * cur_attenuation;
                }
                break;
            case MaterialType::DIELECTRIC:
                if (!rec.mat_ptr->Object->dielectric->Scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                    return emitted * cur_attenuation;
                }
                break;
            case MaterialType::DIFFUSELIGHT:
                emitted = rec.mat_ptr->Object->diffuse_light->Emitted(rec.u, rec.v, rec.p);
                return emitted * cur_attenuation;
            default:
                return background;
            }

            cur_attenuation = attenuation * cur_attenuation;
            cur_ray = scattered;
        }
    }

    return background; // exceeded recursion
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

#define THREADS_PER_BLOCK 256
#if __CUDA_ARCH__ >= 200
#define MY_KERNEL_MAX_THREADS (4 * THREADS_PER_BLOCK)
#else
#define MY_KERNEL_MAX_THREADS THREADS_PER_BLOCK
#endif

__global__ __launch_bounds__(MY_KERNEL_MAX_THREADS) void Kernel(unsigned int* pos, unsigned int width,
                                                                unsigned int height,
                                                                const unsigned int samples_per_pixel,
                                                                const unsigned int max_depth, Hittable* world,
                                                                curandState* rand_state, InputStruct inputs)
{
    // extern __shared__ uchar4 sdata[];
    // Define shared memory for the rand_state array.
    // Each thread in the block will have one curandState element.
    // extern __shared__ curandState shared_rand_state[];

    int x, y, z;
    GetXYZCoords(x, y, z);

    if ((x >= width) || (y >= height))
        return;

    unsigned int pixel_index = (y * width + x);

    // Copy from global to shared memory
    // shared_rand_state[threadIdx.y * blockDim.x + threadIdx.x] = rand_state[pixel_index];
    curandState local_rand_state = rand_state[pixel_index];
    // Make sure all threads have finished copying
    // __syncthreads();
    // curandState local_rand_state = shared_rand_state[threadIdx.y * blockDim.x + threadIdx.x];

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
        Vec3 secondPlanePos = (inputs.far_plane * distFromCenter) + ((1.0f / inputs.fov * 10.0f) * forwardV) + origin;
        Vec3 dirVector = Normalize(secondPlanePos - startPos);

        Ray r = Ray(startPos, dirVector);
        col = col + color(r, world, max_depth, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;

    col = col / (float)(samples_per_pixel);
    col.e[0] = 255.0f * sqrtf(col.x());
    col.e[1] = 255.0f * sqrtf(col.y());
    col.e[2] = 255.0f * sqrtf(col.z());

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
//                        new Material(Vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND,
//                        Mat::metal));
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

extern "C" void LaunchKernel(unsigned int* pos, unsigned int image_width, unsigned int image_height,
                             const unsigned int samples_per_pixel, const unsigned int max_depth, Hittable* world,
                             curandState* d_rand_state, InputStruct inputs)
{
    // Calculate grid size
    dim3 block(16, 16, 1);
    dim3 grid(image_width / block.x, image_height / block.y, 1);
    // Calculate the size of shared memory:
    // number of threads per block * size of each curandState element
    // size_t sbytes = block.x * block.y * sizeof(curandState);

    // HittableList* d_world;
    // cudaMallocManaged((void**)&d_world, world->objects.size() * sizeof(HittableList));

    // for (int i = 0; i < world->objects.size(); i++) {
    //     d_world[i] = world->objects[i];
    // }

    Kernel<<<grid, block>>>(pos, image_width, image_height, samples_per_pixel, max_depth, world, d_rand_state, inputs);
    cudaDeviceSynchronize();

    // cudaFree(d_world);
}

extern "C" void LaunchRandInit(curandState* d_rand_state2)
{
    RandInit<<<1, 1>>>(d_rand_state2);
    cudaDeviceSynchronize();
}

extern "C" void LaunchRenderInit(dim3 grid, dim3 block, unsigned int window_width, unsigned int window_height,
                                 curandState* d_rand_state)
{
    RenderInit<<<grid, block>>>(window_width, window_height, d_rand_state);
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