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
inline __device__ float clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

inline __device__ int clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}

// convert floating point rgb color to 8-bit integer
inline __device__ int rgbToInt(float r, float g, float b)
{
    r = clamp(r, 0.0f, 255.0f);
    g = clamp(g, 0.0f, 255.0f);
    b = clamp(b, 0.0f, 255.0f);
    float a = 255.0f;
    return (int(a) << 24) | (int(b) << 16) | (int(g) << 8) | int(r);
}

// the reverse
inline __device__ float3 intToRgb(int val)
{
    float r = val % 256;
    float g = (val % (256 * 256)) / 256;
    float b = val / (256 * 256);
    return make_float3(r, g, b);
}

inline __device__ bool hit_sphere(const float3& center, float radius, const ray& r)
{
    float3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0f * a * c;

    return (discriminant > 0.0f);
}

inline __device__ float3 color(const ray& r, hittable** world, const int max_depth, curandState* local_rand_state) {
    ray cur_ray = r;
    float3 cur_attenuation = make_float3(1.0, 1.0, 1.0);
    for (int i = 0; i < max_depth; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            float3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation = attenuation * cur_attenuation;
                cur_ray = scattered;
            }
            else {
                return make_float3(0.0, 0.0, 0.0);
            }
        }
        else {
            float3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y + 1.0f);
            float3 c = (1.0f - t) * make_float3(1.0, 1.0, 1.0) + t * make_float3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return make_float3(0.0, 0.0, 0.0); // exceeded recursion
}

inline __device__ void getXYZCoords(int& x, int& y, int& z)
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

__global__ void kernel(unsigned int* pos, unsigned int width, unsigned int height, const unsigned int samples_per_pixel, const unsigned int max_depth, hittable** world, curandState* rand_state, InputStruct inputs)
{
    extern __shared__ uchar4 sdata[];

    int x, y, z;
    getXYZCoords(x, y, z);

    if ((x >= width) || (y >= height))
        return;

    unsigned int pixel_index = (y * width + x);

    curandState local_rand_state = rand_state[pixel_index];

    float3 col = make_float3(0.0f, 0.0f, 0.0f);

    float3 origin = make_float3(inputs.origin_x, inputs.origin_y, inputs.origin_z);
    float3 forwardV = make_float3(inputs.orientation_x, inputs.orientation_y, inputs.orientation_z);
    float3 upV = make_float3(inputs.up_x, inputs.up_y, inputs.up_z);
    float3 rightV = normalize(cross(upV, forwardV));

    float sizeFarPlane = 10;
    float sizeNearPlane = sizeFarPlane * 0.1f;
    float distFarPlane = 10;
    float distFirstPlane = distFarPlane * 0.1f;

    float3 center = make_float3(width / 2.0, height / 2.0, 0.0f);
    float3 distFromCenter = ((x - center.x) / width) * rightV + ((center.y - y) / width) * upV;
    float3 startPos = (sizeNearPlane * distFromCenter) + origin + (distFirstPlane * forwardV);
    float3 secondPlanePos = (sizeFarPlane * distFromCenter) + origin + (distFarPlane * forwardV) + origin;

    float3 dirVector = normalize(secondPlanePos - startPos);

    ray r = ray(startPos, dirVector);
    // col = col + color(r, world, max_depth, &local_rand_state);

    for (int s = 0; s < samples_per_pixel; s++)
    {
        // calculate uv coordinates
    //    float u = (float)(x + curand_uniform(&local_rand_state)) / (float)(width);
    //    float v = (float)(y + curand_uniform(&local_rand_state)) / (float)(height);
       col = col + color(r, world, max_depth, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;

    col = col / (float)(samples_per_pixel);
    col.x = 255.0f * sqrt(col.x);
    col.y = 255.0f * sqrt(col.y);
    col.z = 255.0f * sqrt(col.z);

    // write output vertex
    pos[pixel_index] = rgbToInt(col.x, col.y, col.z);
}

__global__ void rand_init(curandState* rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        curand_init(1984, 0, 0, rand_state);
}

__global__ void render_init(unsigned int window_width, unsigned int window_height, curandState* rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= window_width) || (j >= window_height))
        return;

    unsigned int pixel_index = (j * window_width + i);
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void create_world(hittable** d_list, hittable** d_world, const float aspect_ratio, curandState* rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;

        int i = 0;

        //d_list[i++] = new sphere(make_float3(0, -1000.0, -1), 1000,
        //    new lambertian(make_float3(0.5, 0.5, 0.5)));
        //for (int a = -11; a < 11; a++) {
        //    for (int b = -11; b < 11; b++) {
        //        float choose_mat = RND;
        //        float3 center = make_float3(a + RND, 0.2, b + RND);
        //        if (choose_mat < 0.8f) {
        //            d_list[i++] = new sphere(center, 0.2,
        //                new lambertian(make_float3(RND * RND, RND * RND, RND * RND)));
        //        }
        //        else if (choose_mat < 0.95f) {
        //            d_list[i++] = new sphere(center, 0.2,
        //                new metal(make_float3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
        //        }
        //        else {
        //            d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
        //        }
        //    }
        //}
        //d_list[i++] = new sphere(make_float3(0, 1, 0), 1.0, new dielectric(1.5));
        //d_list[i++] = new sphere(make_float3(0, 1, 0), -0.9, new dielectric(1.5));
        //d_list[i++] = new sphere(make_float3(-4, 1, 0), 1.0, new lambertian(make_float3(0.4, 0.2, 0.1)));
        //d_list[i++] = new sphere(make_float3(4, 1, 0), 1.0, new metal(make_float3(0.7, 0.6, 0.5), 0.0));

        d_list[i++] = new sphere(make_float3(0, -100.5, 0), 100, new lambertian(make_float3(0.8, 0.8, 0.0)));
        d_list[i++] = new sphere(make_float3(0, 0, -1), 0.5, new lambertian(make_float3(0.1, 0.2, 0.5)));
        d_list[i++] = new sphere(make_float3(1, 0, -1), 0.5, new metal(make_float3(0.8, 0.6, 0.2), 0.0));
        d_list[i++] = new sphere(make_float3(-1, 0, -1), 0.5, new dielectric(1.5f));
        d_list[i++] = new sphere(make_float3(-1, 0, -1), -0.45, new dielectric(1.5f));

        *rand_state = local_rand_state;
        *d_world = new hittable_list(d_list, i);
    }
}

__global__ void free_world(hittable** d_list, hittable** d_world, const unsigned int num_hittables) {
    for (int i = 0; i < num_hittables; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* d_world;
}

extern "C"
void launch_kernel(dim3 grid, dim3 block, int sbytes, unsigned int* pos, unsigned int window_width, unsigned int window_height, const unsigned int samples_per_pixel, const unsigned int max_depth, hittable * *d_world, curandState * d_rand_state, InputStruct inputs)
{
    kernel << < grid, block, sbytes >> > (pos, window_width, window_height, samples_per_pixel, max_depth, d_world, d_rand_state, inputs);
    cudaDeviceSynchronize();
}

extern "C"
void launch_rand_init(curandState * d_rand_state2)
{
    rand_init << < 1, 1 >> > (d_rand_state2);
    cudaDeviceSynchronize();
}

extern "C"
void launch_render_init(dim3 grid, dim3 block, unsigned int window_width, unsigned int window_height, curandState * d_rand_state)
{
    render_init << < grid, block >> > (window_width, window_height, d_rand_state);
    cudaDeviceSynchronize();
}

extern "C"
void launch_create_world(hittable * *d_list, hittable * *d_world, const float aspect_ratio, curandState * d_rand_state2)
{
    create_world << < 1, 1 >> > (d_list, d_world, aspect_ratio, d_rand_state2);
    cudaDeviceSynchronize();
}

extern "C"
void launch_free_world(hittable * *d_list, hittable * *d_world, const unsigned int num_hittables)
{
    free_world << < 1, 1 >> > (d_list, d_world, num_hittables);
    cudaDeviceSynchronize();
}

// __global__ void
// cudaRender(unsigned int *g_odata, int imgw)
// {
//     extern __shared__ uchar4 sdata[];

//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int bw = blockDim.x;
//     int bh = blockDim.y;
//     int x = blockIdx.x * bw + tx;
//     int y = blockIdx.y * bh + ty;

//     unsigned int pixel_index = (y * imgw + x);

//     uchar4 c4 = make_uchar4((x & 0x20) ? 100 : 0, 0, (y & 0x20) ? 100 : 0, 0);
//     g_odata[pixel_index] = rgbToInt(c4.z, c4.y, c4.x);
// }

// extern "C" void
// launch_cudaRender(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int imgw)
// {
//     cudaRender<<<grid, block, sbytes>>>(g_odata, imgw);
// }