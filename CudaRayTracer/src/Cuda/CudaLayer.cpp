#include "Cuda/CudaLayer.h"
#include "Core/Application.h"

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui.h>

#include "Utils/RawStbImage.h"

#include "ImGui/ImGuiFileDialog.h"

extern "C" void LaunchKernel(unsigned int* pos, unsigned int image_width, unsigned int image_height,
                             const unsigned int samples_per_pixel, const unsigned int max_depth, Hittable* world,
                             curandState* d_rand_state, InputStruct inputs);

extern "C" void LaunchRandInit(curandState* d_rand_state2);

extern "C" void LaunchRenderInit(dim3 grid, dim3 block, unsigned int image_width, unsigned int image_height,
                                 curandState* d_rand_state);

static ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_DefaultOpen;

CudaLayer::CudaLayer() : Layer("CudaLayer")
{
}

void CudaLayer::OnAttach()
{
    findCudaDevice();

    size_t pValue;
    cudaDeviceSetLimit(cudaLimitStackSize, 4096);
    cudaDeviceGetLimit(&pValue, cudaLimitStackSize);
    RT_INFO("CUDA Stack Size Limit: {0} bytes", pValue);

    InitCudaBuffers();
    InitGLBuffers();

    RunCudaInit();

    GenerateWorld();

    m_Camera = std::make_unique<Camera>(m_ImageWidth, m_ImageHeight, glm::vec3(0.0f, 2.0f, 10.0f));

    glm::vec3 rightV = glm::normalize(glm::cross(m_Camera->m_Orientation, m_Camera->m_Up));
    glm::vec3 upV = glm::normalize(glm::cross(m_Camera->m_Orientation, rightV));

    m_Inputs.origin_x = m_Camera->m_Position.x;
    m_Inputs.origin_y = m_Camera->m_Position.y;
    m_Inputs.origin_z = m_Camera->m_Position.z;

    m_Inputs.orientation_x = m_Camera->m_Orientation.x;
    m_Inputs.orientation_y = m_Camera->m_Orientation.y;
    m_Inputs.orientation_z = m_Camera->m_Orientation.z;

    m_Inputs.up_x = upV.x;
    m_Inputs.up_y = upV.y;
    m_Inputs.up_z = upV.z;

    m_Inputs.far_plane = m_Camera->m_FarPlane;
    m_Inputs.near_plane = m_Camera->m_NearPlane;
    m_Inputs.fov = glm::radians(m_Camera->m_Fov);
}

void CudaLayer::InitCudaBuffers()
{
    checkCudaErrors(cudaMalloc(&m_CudaDevRenderBuffer, m_SizeTexData)); // Allocate CUDA memory for color output
    // Allocate random state
    checkCudaErrors(cudaMalloc((void**)&m_DrandState, m_NumTexels * sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void**)&m_DrandState2, 1 * sizeof(curandState)));
}

void CudaLayer::InitGLBuffers()
{
    // create an OpenGL texture
    glGenTextures(1, &m_Texture);            // generate 1 texture
    glBindTexture(GL_TEXTURE_2D, m_Texture); // set it as current target
    // set basic texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // Specify 2D texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_ImageWidth, m_ImageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    // Register this texture with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterImage(&m_CudaTexResource, m_Texture, GL_TEXTURE_2D,
                                                cudaGraphicsRegisterFlagsWriteDiscard));
}

void CudaLayer::RunCudaInit()
{
    LaunchRandInit(m_DrandState2);

    dim3 block(16, 16, 1);
    dim3 grid(m_ImageWidth / block.x, m_ImageHeight / block.y, 1);

    LaunchRenderInit(grid, block, m_ImageWidth, m_ImageHeight, m_DrandState);
}

void CudaLayer::GenerateWorld()
{
    m_ListSize = 17;
    // Coalesced memory
    // Calculate total size of memory needed
    m_LambertianSize = sizeof(Material) + sizeof(Material::ObjectUnion) + sizeof(Lambertian);
    m_MetalSize = sizeof(Material) + sizeof(Material::ObjectUnion) + sizeof(Metal);
    m_DielectricSize = sizeof(Material) + sizeof(Material::ObjectUnion) + sizeof(Dielectric);
    m_DiffuseSize = sizeof(Material) + sizeof(Material::ObjectUnion) + sizeof(DiffuseLight);
    m_ConstantSize = sizeof(Texture) + sizeof(Texture::ObjectUnion) + sizeof(Constant);
    m_CheckerSize = sizeof(Texture) + sizeof(Texture::ObjectUnion) + sizeof(Checker) + 2 * sizeof(Constant);
    // m_TextureImageFilename = "assets/textures/industrial_sunset_puresky.jpg";
    m_TextureImageData = LoadImage(m_TextureImageFilename, m_TextureImageData, &m_TextureImageWidth,
                                   &m_TextureImageHeight, &m_TextureImageNR);
    m_ImageSize = sizeof(Texture) + sizeof(Texture::ObjectUnion) + sizeof(Image) +
                  m_TextureImageWidth * m_TextureImageHeight * m_TextureImageNR * sizeof(unsigned char);
    m_SphereSize = sizeof(Hittable) + sizeof(Hittable::ObjectUnion) + sizeof(Sphere);
    m_XYrectSize = sizeof(Hittable) + sizeof(Hittable::ObjectUnion) + sizeof(XYRect);
    m_XZrectSize = sizeof(Hittable) + sizeof(Hittable::ObjectUnion) + sizeof(XZRect);
    m_YZrectSize = sizeof(Hittable) + sizeof(Hittable::ObjectUnion) + sizeof(YZRect);

    RT_INFO("Constant Size = {0}", m_ConstantSize);
    RT_INFO("Checker Size = {0}", m_CheckerSize);
    RT_INFO("Image Size = {0}", m_ImageSize);

    m_GroundSize = m_XZrectSize + m_LambertianSize + m_CheckerSize;
    m_SkyboxSize = m_SphereSize + m_LambertianSize + m_ImageSize;
    m_SpheresSize = m_SphereSize + m_MetalSize + m_CheckerSize;

    m_TotalSize = (m_ListSize * sizeof(Hittable*)) + m_GroundSize + ((m_ListSize - 1) * m_SpheresSize);
    m_TotalWorldSize = sizeof(Hittable) + sizeof(Hittable::ObjectUnion) + sizeof(BVHNode);

    // Allocate the memory
    checkCudaErrors(cudaMallocManaged(&m_ListMemory, m_TotalSize));

    m_List = (Hittable**)m_ListMemory;

    // Ground XZRect
    // Partitioning
    char* basePtr = m_ListMemory + m_ListSize * sizeof(Hittable*);
    m_List[0] = (Hittable*)(basePtr);
    m_List[0]->Object = (Hittable::ObjectUnion*)(m_List[0] + 1);
    m_List[0]->Object->xz_rect = (XZRect*)(m_List[0]->Object + 1);
    m_List[0]->Object->xz_rect->mat_ptr = (Material*)(m_List[0]->Object->xz_rect + 1);
    m_List[0]->Object->xz_rect->mat_ptr->Object = (Material::ObjectUnion*)(m_List[0]->Object->xz_rect->mat_ptr + 1);
    m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian =
        (Lambertian*)(m_List[0]->Object->xz_rect->mat_ptr->Object + 1);
    m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo =
        (Texture*)(m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian + 1);
    m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object =
        (Texture::ObjectUnion*)(m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo + 1);
    m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->checker =
        (Checker*)(m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object + 1);
    m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->checker->odd =
        (Constant*)(m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->checker + 1);
    m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->checker->even =
        (Constant*)(m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->checker->odd + 1);

    m_List[0]->type = HittableType::XZRECT;
    m_List[0]->isActive = true;

    new (m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->checker->odd)
        Constant(Vec3(0.2f, 0.3f, 0.1f));
    new (m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->checker->even)
        Constant(Vec3(0.9f, 0.9f, 0.9f));
    new (m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->checker)
        Checker(m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->checker->odd,
                m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->checker->even);
    m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->type = TextureType::CHECKER;
    new (m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian)
        Lambertian(m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo);
    // Set the type of the Material after constructing it, so the assignment won't be overwritten.
    m_List[0]->Object->xz_rect->mat_ptr->type = MaterialType::LAMBERTIAN;
    m_List[0]->Object->xz_rect = new (m_List[0]->Object->xz_rect)
        XZRect(Vec3(0.0f, -0.5f, 0.0f), 1000.0f, 1000.0f, m_List[0]->Object->xz_rect->mat_ptr);

    // Skybox Sphere
    // Partitioning
    // char* basePtr1 = m_ListMemory + m_ListSize * sizeof(Hittable*) + groundSize;
    // m_List[1] = (Hittable*)(basePtr1);
    // m_List[1]->Object = (Hittable::ObjectUnion*)(m_List[1] + 1);
    // m_List[1]->Object->sphere = (Sphere*)(m_List[1]->Object + 1);
    // m_List[1]->Object->sphere->mat_ptr = (Material*)(m_List[1]->Object->sphere + 1);
    // m_List[1]->Object->sphere->mat_ptr->Object = (Material::ObjectUnion*)(m_List[1]->Object->sphere->mat_ptr + 1);
    // m_List[1]->Object->sphere->mat_ptr->Object->lambertian =
    //     (Lambertian*)(m_List[1]->Object->sphere->mat_ptr->Object + 1);
    // m_List[1]->Object->sphere->mat_ptr->Object->lambertian->albedo =
    //     (Texture*)(m_List[1]->Object->sphere->mat_ptr->Object->lambertian + 1);
    // m_List[1]->Object->sphere->mat_ptr->Object->lambertian->albedo->Object =
    //     (Texture::ObjectUnion*)(m_List[1]->Object->sphere->mat_ptr->Object->lambertian->albedo + 1);
    // m_List[1]->Object->sphere->mat_ptr->Object->lambertian->albedo->Object->image =
    //     (Image*)(m_List[1]->Object->sphere->mat_ptr->Object->lambertian->albedo->Object + 1);
    // m_List[1]->Object->sphere->mat_ptr->Object->lambertian->albedo->Object->image->data =
    //     (unsigned char*)(m_List[1]->Object->sphere->mat_ptr->Object->lambertian->albedo->Object->image + 1);

    // m_List[1]->type = HittableType::SPHERE;

    // checkCudaErrors(cudaMemcpy(
    //     m_List[1]->Object->sphere->mat_ptr->Object->lambertian->albedo->Object->image->data, m_TextureImageData,
    //     m_TextureImageWidth * m_TextureImageHeight * m_TextureImageNR * sizeof(unsigned char),
    //     cudaMemcpyHostToDevice));
    // STBI_FREE(m_TextureImageData);

    // new (m_List[1]->Object->sphere->mat_ptr->Object->lambertian->albedo->Object->image)
    //     Image(m_List[1]->Object->sphere->mat_ptr->Object->lambertian->albedo->Object->image->data,
    //     m_TextureImageWidth,
    //           m_TextureImageHeight);
    // m_List[1]->Object->sphere->mat_ptr->Object->lambertian->albedo->type = TextureType::IMAGE;
    // new (m_List[1]->Object->sphere->mat_ptr->Object->lambertian)
    //     Lambertian(m_List[1]->Object->sphere->mat_ptr->Object->lambertian->albedo);
    // // Set the type of the Material after constructing it, so the assignment won't be overwritten.
    // m_List[1]->Object->sphere->mat_ptr->type = MaterialType::LAMBERTIAN;
    // m_List[1]->Object->sphere =
    //     new (m_List[1]->Object->sphere) Sphere(Vec3(0.0f, 0.0f, 0.0f), 1000.0f, m_List[1]->Object->sphere->mat_ptr);

    // Spheres
    int i = 1;
    for (int a = -2; a < 2; a++) {
        for (int b = -2; b < 2; b++) {
            // Partitioning
            char* basePtr = m_ListMemory + m_ListSize * sizeof(Hittable*) + m_GroundSize + (i - 1) * m_SpheresSize;
            m_List[i] = (Hittable*)(basePtr);
            m_List[i]->Object = (Hittable::ObjectUnion*)(m_List[i] + 1);
            m_List[i]->Object->sphere = (Sphere*)(m_List[i]->Object + 1);
            m_List[i]->Object->sphere->mat_ptr = (Material*)(m_List[i]->Object->sphere + 1);
            m_List[i]->Object->sphere->mat_ptr->Object =
                (Material::ObjectUnion*)(m_List[i]->Object->sphere->mat_ptr + 1);
            m_List[i]->Object->sphere->mat_ptr->Object->metal =
                (Metal*)(m_List[i]->Object->sphere->mat_ptr->Object + 1);
            m_List[i]->Object->sphere->mat_ptr->Object->metal->albedo =
                (Texture*)(m_List[i]->Object->sphere->mat_ptr->Object->metal + 1);
            m_List[i]->Object->sphere->mat_ptr->Object->metal->albedo->Object =
                (Texture::ObjectUnion*)(m_List[i]->Object->sphere->mat_ptr->Object->metal->albedo + 1);
            m_List[i]->Object->sphere->mat_ptr->Object->metal->albedo->Object->checker =
                (Checker*)(m_List[i]->Object->sphere->mat_ptr->Object->metal->albedo->Object + 1);
            m_List[i]->Object->sphere->mat_ptr->Object->metal->albedo->Object->checker->odd =
                (Constant*)(m_List[i]->Object->sphere->mat_ptr->Object->metal->albedo->Object->checker + 1);
            m_List[i]->Object->sphere->mat_ptr->Object->metal->albedo->Object->checker->even =
                (Constant*)(m_List[i]->Object->sphere->mat_ptr->Object->metal->albedo->Object->checker->odd + 1);

            m_List[i]->type = HittableType::SPHERE;
            m_List[i]->isActive = true;

            float choose_mat = RND;
            Vec3 center = Vec3(a + RND, 0.0, b + RND);

            if (choose_mat < 0.5f) {
                new (m_List[i]->Object->sphere->mat_ptr->Object->lambertian->albedo->Object->constant)
                    Constant(Vec3(RND * RND, RND * RND, RND * RND));
                m_List[i]->Object->sphere->mat_ptr->Object->lambertian->albedo->type = TextureType::CONSTANT;
                new (m_List[i]->Object->sphere->mat_ptr->Object->lambertian)
                    Lambertian(m_List[i]->Object->sphere->mat_ptr->Object->lambertian->albedo);
                // Set the type of the Material after constructing it, so the assignment won't be overwritten.
                m_List[i]->Object->sphere->mat_ptr->type = MaterialType::LAMBERTIAN;
                m_List[i]->Object->sphere =
                    new (m_List[i]->Object->sphere) Sphere(center, 0.2f, m_List[i]->Object->sphere->mat_ptr);
            }
            else if (choose_mat < 0.95f) {
                new (m_List[i]->Object->sphere->mat_ptr->Object->metal->albedo->Object->constant)
                    Constant(Vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)));
                m_List[i]->Object->sphere->mat_ptr->Object->metal->albedo->type = TextureType::CONSTANT;
                new (m_List[i]->Object->sphere->mat_ptr->Object->metal)
                    Metal(m_List[i]->Object->sphere->mat_ptr->Object->metal->albedo, 0.5f * RND);
                // Set the type of the Material after constructing it, so the assignment won't be overwritten.
                m_List[i]->Object->sphere->mat_ptr->type = MaterialType::METAL;
                m_List[i]->Object->sphere =
                    new (m_List[i]->Object->sphere) Sphere(center, 0.2f, m_List[i]->Object->sphere->mat_ptr);
            }
            else {
                new (m_List[i]->Object->sphere->mat_ptr->Object->diffuse_light->albedo->Object->constant)
                    Constant(Vec3(1.0f, 1.0f, 1.0f));
                m_List[i]->Object->sphere->mat_ptr->Object->diffuse_light->albedo->type = TextureType::CONSTANT;
                new (m_List[i]->Object->sphere->mat_ptr->Object->diffuse_light)
                    DiffuseLight(m_List[i]->Object->sphere->mat_ptr->Object->diffuse_light->albedo, 3);
                // Set the type of the Material after constructing it, so the assignment won't be overwritten.
                m_List[i]->Object->sphere->mat_ptr->type = MaterialType::DIFFUSELIGHT;
                m_List[i]->Object->sphere =
                    new (m_List[i]->Object->sphere) Sphere(center, 0.5f, m_List[i]->Object->sphere->mat_ptr);
            }
            i++;
        }
    }

    checkCudaErrors(cudaMallocManaged(&m_WorldMemory, m_TotalWorldSize));
    // Partition the memory
    char* worldBasePtr = m_WorldMemory;
    m_World = (Hittable*)worldBasePtr;
    m_World->Object = (Hittable::ObjectUnion*)(m_World + 1);
    m_World->Object->bvh_node = (BVHNode*)(m_World->Object + 1);

    // Initialize the objects
    m_World->type = HittableType::BVHNODE;
    m_World->Object->bvh_node = new (m_World->Object->bvh_node) BVHNode(m_List, 0, m_ListSize);

    // Normal Allocations
    // checkCudaErrors(cudaMallocManaged(&m_List, sizeof(Hittable*)));

    // // Ground XZRect
    // checkCudaErrors(cudaMallocManaged(&m_List[0], sizeof(Hittable)));
    // checkCudaErrors(cudaMallocManaged(&m_List[0]->Object, sizeof(Hittable::ObjectUnion)));
    // checkCudaErrors(cudaMallocManaged(&m_List[0]->Object->xz_rect, sizeof(XZRECT)));
    // checkCudaErrors(cudaMallocManaged(&m_List[0]->Object->xz_rect->mat_ptr, sizeof(Material)));
    // checkCudaErrors(cudaMallocManaged(&m_List[0]->Object->xz_rect->mat_ptr->Object, sizeof(Material::ObjectUnion)));
    // checkCudaErrors(cudaMallocManaged(&m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian, sizeof(Lambertian)));
    // checkCudaErrors(
    //     cudaMallocManaged(&m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo, sizeof(Texture)));
    // checkCudaErrors(cudaMallocManaged(&m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object,
    //                                   sizeof(Texture::ObjectUnion)));
    // checkCudaErrors(cudaMallocManaged(&m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->checker,
    //                                   sizeof(Checker)));
    // checkCudaErrors(cudaMallocManaged(
    //     &m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->checker->odd, sizeof(Constant)));
    // checkCudaErrors(cudaMallocManaged(
    //     &m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->checker->even, sizeof(Constant)));

    // m_List[0]->type = HittableType::XZRECT;

    // new (m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->checker->odd)
    //     Constant(Vec3(0.2f, 0.3f, 0.1f));
    // new (m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->checker->even)
    //     Constant(Vec3(0.9f, 0.9f, 0.9f));
    // new (m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->checker)
    //     Checker(m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->checker->odd,
    //             m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->checker->even);
    // m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo->type = TextureType::CHECKER;
    // new (m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian)
    //     Lambertian(m_List[0]->Object->xz_rect->mat_ptr->Object->lambertian->albedo);
    // // Set the type of the Material after constructing it, so the assignment won't be overwritten.
    // m_List[0]->Object->xz_rect->mat_ptr->type = MaterialType::LAMBERTIAN;
    // m_List[0]->Object->xz_rect = new (m_List[0]->Object->xz_rect)
    //     XZRect(Vec3(0.0f, -0.5f, 0.0f), 1000.0f, 1000.0f, m_List[0]->Object->xz_rect->mat_ptr);

    // int i = 1;
    // for (int a = -2; a < 2; a++) {
    //     for (int b = -2; b < 2; b++) {
    //         checkCudaErrors(cudaMallocManaged(&m_List[i], sizeof(Hittable)));
    //         checkCudaErrors(cudaMallocManaged(&m_List[i]->Object, sizeof(Hittable::ObjectUnion)));
    //         checkCudaErrors(cudaMallocManaged(&m_List[i]->Object->sphere, sizeof(Sphere)));
    //         checkCudaErrors(cudaMallocManaged(&m_List[i]->Object->sphere->mat_ptr, sizeof(Material)));
    //         checkCudaErrors(
    //             cudaMallocManaged(&m_List[i]->Object->sphere->mat_ptr->Object, sizeof(Material::ObjectUnion)));
    //         checkCudaErrors(
    //             cudaMallocManaged(&m_List[i]->Object->sphere->mat_ptr->Object->lambertian, sizeof(Lambertian)));
    //         checkCudaErrors(
    //             cudaMallocManaged(&m_List[i]->Object->sphere->mat_ptr->Object->lambertian->albedo, sizeof(Texture)));
    //         checkCudaErrors(cudaMallocManaged(&m_List[i]->Object->sphere->mat_ptr->Object->lambertian->albedo->Object,
    //                                           sizeof(Texture::ObjectUnion)));
    //         checkCudaErrors(cudaMallocManaged(
    //             &m_List[i]->Object->sphere->mat_ptr->Object->lambertian->albedo->Object->constant,
    //             sizeof(Constant)));

    //         float choose_mat = RND;
    //         Vec3 center = Vec3(a + RND, 0.0, b + RND);

    //         m_List[i]->type = HittableType::SPHERE;

    //         if (choose_mat < 0.5f) {
    //             new (m_List[i]->Object->sphere->mat_ptr->Object->lambertian->albedo->Object->constant)
    //                 Constant(Vec3(RND * RND, RND * RND, RND * RND));
    //             m_List[i]->Object->sphere->mat_ptr->Object->lambertian->albedo->type = TextureType::CONSTANT;
    //             new (m_List[i]->Object->sphere->mat_ptr->Object->lambertian)
    //                 Lambertian(m_List[i]->Object->sphere->mat_ptr->Object->lambertian->albedo);
    //             // Set the type of the Material after constructing it, so the assignment won't be overwritten.
    //             m_List[i]->Object->sphere->mat_ptr->type = MaterialType::LAMBERTIAN;
    //             m_List[i]->Object->sphere =
    //                 new (m_List[i]->Object->sphere) Sphere(center, 0.2f, m_List[i]->Object->sphere->mat_ptr);
    //         }
    //         else if (choose_mat < 0.95f) {
    //             new (m_List[i]->Object->sphere->mat_ptr->Object->metal->albedo->Object->constant)
    //                 Constant(Vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)));
    //             m_List[i]->Object->sphere->mat_ptr->Object->metal->albedo->type = TextureType::CONSTANT;
    //             new (m_List[i]->Object->sphere->mat_ptr->Object->metal)
    //                 Metal(m_List[i]->Object->sphere->mat_ptr->Object->metal->albedo, 0.5f * RND);
    //             // Set the type of the Material after constructing it, so the assignment won't be overwritten.
    //             m_List[i]->Object->sphere->mat_ptr->type = MaterialType::METAL;
    //             m_List[i]->Object->sphere =
    //                 new (m_List[i]->Object->sphere) Sphere(center, 0.2f, m_List[i]->Object->sphere->mat_ptr);
    //         }
    //         else {
    //             new (m_List[i]->Object->sphere->mat_ptr->Object->diffuse_light->albedo->Object->constant)
    //                 Constant(Vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)));
    //             m_List[i]->Object->sphere->mat_ptr->Object->diffuse_light->albedo->type = TextureType::CONSTANT;
    //             new (m_List[i]->Object->sphere->mat_ptr->Object->diffuse_light)
    //                 DiffuseLight(m_List[i]->Object->sphere->mat_ptr->Object->diffuse_light->albedo, 3);
    //             // Set the type of the Material after constructing it, so the assignment won't be overwritten.
    //             m_List[i]->Object->sphere->mat_ptr->type = MaterialType::DIFFUSELIGHT;
    //             m_List[i]->Object->sphere =
    //                 new (m_List[i]->Object->sphere) Sphere(center, 0.5f, m_List[i]->Object->sphere->mat_ptr);
    //         }
    //         i++;
    //     }
    // }

    // checkCudaErrors(cudaMallocManaged(&m_World, sizeof(Hittable)));
    // checkCudaErrors(cudaMallocManaged(&m_World->Object, sizeof(Hittable::ObjectUnion)));
    // checkCudaErrors(cudaMallocManaged(&m_World->Object->bvh_node, sizeof(BVHNode)));
    // m_World->type = HittableType::BVHNODE;
    // m_World->Object->bvh_node = new (m_World->Object->bvh_node) BVHNode(m_List, 0, m_ListSize);

    // Hittable* ground;
    // checkCudaErrors(cudaMallocManaged(&ground, sizeof(Hittable)));
    // checkCudaErrors(cudaMallocManaged(&ground->mat_ptr, sizeof(Material)));
    // checkCudaErrors(cudaMallocManaged(&ground->mat_ptr->albedo,
    // sizeof(Texture)));
    // checkCudaErrors(cudaMallocManaged(&ground->mat_ptr->albedo->odd,
    // sizeof(Texture)));
    // checkCudaErrors(cudaMallocManaged(&ground->mat_ptr->albedo->even,
    // sizeof(Texture))); m_World->Add(new(ground) Hittable(Vec3(0.0f, -0.5f,
    // 0.0f), 1000.0f, 1000.0f, new(ground->mat_ptr)
    // Material(new(ground->mat_ptr->albedo)
    // Texture(new(ground->mat_ptr->albedo->odd) Texture(Vec3(0.2f, 0.3f, 0.1f),
    // Tex::constant_texture), new(ground->mat_ptr->albedo->even)
    // Texture(Vec3(0.9f, 0.9f, 0.9f), Tex::constant_texture),
    // Tex::checker_texture), Mat::lambertian), Hitt::xz_rect));

    // Hittable* skybox_sphere;
    // checkCudaErrors(cudaMallocManaged(&skybox_sphere, sizeof(Hittable)));
    // checkCudaErrors(cudaMallocManaged(&skybox_sphere->mat_ptr,
    // sizeof(Material)));
    // checkCudaErrors(cudaMallocManaged(&skybox_sphere->mat_ptr->albedo,
    // sizeof(Texture)));
    // checkCudaErrors(cudaMallocManaged(&skybox_sphere->mat_ptr->albedo->odd,
    // sizeof(Texture)));
    // checkCudaErrors(cudaMallocManaged(&skybox_sphere->mat_ptr->albedo->even,
    // sizeof(Texture))); m_TextureImageFilename =
    // "assets/textures/industrial_sunset_puresky.jpg"; m_TextureImageData =
    // LoadImage(m_TextureImageFilename, m_TextureImageData,
    // &m_TextureImageWidth, &m_TextureImageHeight, &m_TextureImageNR);
    // checkCudaErrors(cudaMallocManaged(&skybox_sphere->mat_ptr->albedo->data,
    // m_TextureImageWidth * m_TextureImageHeight * m_TextureImageNR *
    // sizeof(unsigned char)));
    // checkCudaErrors(cudaMemcpy(skybox_sphere->mat_ptr->albedo->data,
    // m_TextureImageData, m_TextureImageWidth * m_TextureImageHeight *
    // m_TextureImageNR * sizeof(unsigned char), cudaMemcpyHostToDevice));
    // STBI_FREE(m_TextureImageData);
    // m_World->Add(new(skybox_sphere) Hittable(Vec3(0.0f, 0.0f, 0.0f), 1000.0f,
    // new(skybox_sphere->mat_ptr) Material(new(skybox_sphere->mat_ptr->albedo)
    // Texture(skybox_sphere->mat_ptr->albedo->data, m_TextureImageWidth,
    // m_TextureImageHeight, Tex::image_texture), Mat::lambertian),
    // Hitt::sphere));
    // // don't forget to set the path for the object
    // skybox_sphere->mat_ptr->albedo->path = m_TextureImageFilename;

    // Hittable* sphere1;
    // checkCudaErrors(cudaMallocManaged(&sphere1, sizeof(Hittable)));
    // checkCudaErrors(cudaMallocManaged(&sphere1->mat_ptr, sizeof(Material)));
    // checkCudaErrors(cudaMallocManaged(&sphere1->mat_ptr->albedo,
    // sizeof(Texture)));
    // checkCudaErrors(cudaMallocManaged(&sphere1->mat_ptr->albedo->odd,
    // sizeof(Texture)));
    // checkCudaErrors(cudaMallocManaged(&sphere1->mat_ptr->albedo->even,
    // sizeof(Texture))); m_World->Add(new(sphere1) Hittable(Vec3(0.0f, 0.0f,
    // -1.0f), 0.5f, new(sphere1->mat_ptr)
    // Material(new(sphere1->mat_ptr->albedo) Texture(Vec3(0.1f, 0.2f, 0.5f),
    // Tex::constant_texture), Mat::lambertian), Hitt::sphere));

    // Hittable* sphere2;
    // checkCudaErrors(cudaMallocManaged(&sphere2, sizeof(Hittable)));
    // checkCudaErrors(cudaMallocManaged(&sphere2->mat_ptr, sizeof(Material)));
    // checkCudaErrors(cudaMallocManaged(&sphere2->mat_ptr->albedo,
    // sizeof(Texture)));
    // checkCudaErrors(cudaMallocManaged(&sphere2->mat_ptr->albedo->odd,
    // sizeof(Texture)));
    // checkCudaErrors(cudaMallocManaged(&sphere2->mat_ptr->albedo->even,
    // sizeof(Texture))); m_World->Add(new(sphere2) Hittable(Vec3(1.0f, 0.0f,
    // -1.0f), 0.5f, new(sphere2->mat_ptr)
    // Material(new(sphere2->mat_ptr->albedo) Texture(Vec3(0.8f, 0.6f, 0.2f),
    // Tex::constant_texture), 0.0f, Mat::metal), Hitt::sphere));

    // Hittable* glassSphere_a;
    // checkCudaErrors(cudaMallocManaged(&glassSphere_a, sizeof(Hittable)));
    // checkCudaErrors(cudaMallocManaged(&glassSphere_a->mat_ptr,
    // sizeof(Material)));
    // checkCudaErrors(cudaMallocManaged(&glassSphere_a->mat_ptr->albedo,
    // sizeof(Texture)));
    // checkCudaErrors(cudaMallocManaged(&glassSphere_a->mat_ptr->albedo->odd,
    // sizeof(Texture)));
    // checkCudaErrors(cudaMallocManaged(&glassSphere_a->mat_ptr->albedo->even,
    // sizeof(Texture))); m_World->Add(new(glassSphere_a) Hittable(Vec3(-1.0f,
    // 0.0f, -1.0f), 0.5f, new(glassSphere_a->mat_ptr) Material(1.5f,
    // Mat::dielectric), Hitt::sphere));

    // Hittable* glassSphere_b;
    // checkCudaErrors(cudaMallocManaged(&glassSphere_b, sizeof(Hittable)));
    // checkCudaErrors(cudaMallocManaged(&glassSphere_b->mat_ptr,
    // sizeof(Material)));
    // checkCudaErrors(cudaMallocManaged(&glassSphere_b->mat_ptr->albedo,
    // sizeof(Texture)));
    // checkCudaErrors(cudaMallocManaged(&glassSphere_b->mat_ptr->albedo->odd,
    // sizeof(Texture)));
    // checkCudaErrors(cudaMallocManaged(&glassSphere_b->mat_ptr->albedo->even,
    // sizeof(Texture))); m_World->Add(new(glassSphere_b) Hittable(Vec3(-1.0f,
    // 0.0f, -1.0f), -0.45f, new(glassSphere_b->mat_ptr) Material(1.5f,
    // Mat::dielectric), Hitt::sphere));

    // Hittable* light_sphere;
    // checkCudaErrors(cudaMallocManaged(&light_sphere, sizeof(Hittable)));
    // checkCudaErrors(cudaMallocManaged(&light_sphere->mat_ptr,
    // sizeof(Material)));
    // checkCudaErrors(cudaMallocManaged(&light_sphere->mat_ptr->albedo,
    // sizeof(Texture)));
    // checkCudaErrors(cudaMallocManaged(&light_sphere->mat_ptr->albedo->odd,
    // sizeof(Texture)));
    // checkCudaErrors(cudaMallocManaged(&light_sphere->mat_ptr->albedo->even,
    // sizeof(Texture))); m_TextureImageFilename = "assets/textures/8k_sun.jpg";
    // m_TextureImageData = LoadImage(m_TextureImageFilename,
    // m_TextureImageData, &m_TextureImageWidth, &m_TextureImageHeight,
    // &m_TextureImageNR);
    // checkCudaErrors(cudaMallocManaged(&light_sphere->mat_ptr->albedo->data,
    // m_TextureImageWidth * m_TextureImageHeight * m_TextureImageNR *
    // sizeof(unsigned char)));
    // checkCudaErrors(cudaMemcpy(light_sphere->mat_ptr->albedo->data,
    // m_TextureImageData, m_TextureImageWidth * m_TextureImageHeight *
    // m_TextureImageNR * sizeof(unsigned char), cudaMemcpyHostToDevice));
    // STBI_FREE(m_TextureImageData);
    // m_World->Add(new(light_sphere) Hittable(Vec3(0.0f, 2.0f, 0.0f), 0.5f,
    // new(light_sphere->mat_ptr) Material(new(light_sphere->mat_ptr->albedo)
    // Texture(light_sphere->mat_ptr->albedo->data, m_TextureImageWidth,
    // m_TextureImageHeight, Tex::image_texture), m_LightIntensity,
    // Mat::diffuse_light), Hitt::sphere));
    // // don't forget to set the path for the object
    // light_sphere->mat_ptr->albedo->path = m_TextureImageFilename;

    // Hittable* rect;
    // checkCudaErrors(cudaMallocManaged(&rect, sizeof(Hittable)));
    // checkCudaErrors(cudaMallocManaged(&rect->mat_ptr, sizeof(Material)));
    // checkCudaErrors(cudaMallocManaged(&rect->mat_ptr->albedo,
    // sizeof(Texture)));
    // checkCudaErrors(cudaMallocManaged(&rect->mat_ptr->albedo->odd,
    // sizeof(Texture)));
    // checkCudaErrors(cudaMallocManaged(&rect->mat_ptr->albedo->even,
    // sizeof(Texture))); m_World->Add(new(rect) Hittable(Vec3(0.0f, 1.0f,
    // -3.0f), 6.0f, 3.0f, new(rect->mat_ptr)
    // Material(new(rect->mat_ptr->albedo) Texture(Vec3(1.0f, 0.0f, 0.0f),
    // Tex::constant_texture), 7, Mat::diffuse_light), Hitt::xy_rect));
}

void CudaLayer::OnUpdate()
{
    Application& app = Application::Get();
    if (!app.GetWindow().m_PauseRender) {
        RunCudaUpdate();
    }
}

void CudaLayer::RunCudaUpdate()
{
    LaunchKernel(static_cast<unsigned int*>(m_CudaDevRenderBuffer), m_ImageWidth, m_ImageHeight, m_SamplesPerPixel,
                 m_MaxDepth, m_World, m_DrandState, m_Inputs);

    // We want to copy cuda_dev_render_buffer data to the texture.
    // Map buffer objects to get CUDA device pointers.
    cudaArray* texture_ptr;
    checkCudaErrors(cudaGraphicsMapResources(1, &m_CudaTexResource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, m_CudaTexResource, 0, 0));

    checkCudaErrors(cudaMemcpy2DToArray(texture_ptr, 0, 0, m_CudaDevRenderBuffer, m_ImageWidth * sizeof(unsigned int),
                                        m_ImageWidth * sizeof(unsigned int), m_ImageHeight, cudaMemcpyDeviceToDevice));
    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, &m_CudaTexResource, 0));
}

void CudaLayer::OnImGuiRender()
{
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Generated Image");

    if (OnImGuiResize() == false) {
        ImGui::End();
        ImGui::PopStyleVar();
        return;
    }

    ImGui::ImageButton((void*)(intptr_t)m_Texture, ImVec2(m_ImageWidth, m_ImageHeight), ImVec2(0, 1), ImVec2(1, 0), 0);
    ImGui::PopStyleVar();

    if (ImGui::IsWindowHovered()) {
        ImGuiIO& io = ImGui::GetIO();
        if (io.MouseWheel) {
            m_Camera->ProcessMouseScroll(io.MouseWheel);
            m_Inputs.fov = glm::radians(m_Camera->m_Fov);
        }
    }

    if (ImGui::IsWindowFocused()) {

        m_Camera->Inputs((GLFWwindow*)ImGui::GetMainViewport()->PlatformHandle);

        glm::vec3 rightV = glm::normalize(glm::cross(m_Camera->m_Orientation, m_Camera->m_Up));
        glm::vec3 upV = glm::normalize(glm::cross(m_Camera->m_Orientation, rightV));

        m_Inputs.origin_x = m_Camera->m_Position.x;
        m_Inputs.origin_y = m_Camera->m_Position.y;
        m_Inputs.origin_z = m_Camera->m_Position.z;

        m_Inputs.orientation_x = m_Camera->m_Orientation.x;
        m_Inputs.orientation_y = m_Camera->m_Orientation.y;
        m_Inputs.orientation_z = m_Camera->m_Orientation.z;

        m_Inputs.up_x = upV.x;
        m_Inputs.up_y = upV.y;
        m_Inputs.up_z = upV.z;
    }
    else {
        glfwSetInputMode((GLFWwindow*)ImGui::GetMainViewport()->PlatformHandle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        m_Camera->m_FirstClick = true;
    }

    ImGui::End();

    ImGui::Begin("Metrics");
    ImGuiIO& io = ImGui::GetIO();

    ImGui::Text("Dear ImGui %s", ImGui::GetVersion());

    ImGui::Text("Generated image dimensions: %dx%d", m_ImageWidth, m_ImageHeight);

#ifdef RT_DEBUG
    ImGui::Text("Running on Debug mode");
#elif RT_RELEASE
    ImGui::Text("Running on Release mode");
#elif RT_DIST
    ImGui::Text("Running on Dist mode");
#endif

    ImGui::Text("Application average\n %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

    ImGui::End();

    ImGui::Begin("Scene");
    ImGui::Separator();

    if (ImGui::CollapsingHeader("Hittables Settings", base_flags)) {
        for (size_t i = 0; i < m_ListSize; i++) {
            if (m_List[i]->isActive &&
                ImGui::TreeNodeEx((GetTextForEnum(m_List[i]->type) + std::to_string(i)).c_str())) {
                // Hittable switch-case
                switch (m_List[i]->type) {
                case HittableType::SPHERE:
                    if (ImGui::DragFloat3("Position", (float*)&m_List[i]->Object->sphere->center, 0.01f, -FLT_MAX,
                                          FLT_MAX, "%.2f")) {
                        m_World->Object->bvh_node->Destroy();
                        m_World->Object->bvh_node = new (m_World->Object->bvh_node) BVHNode(m_List, 0, m_ListSize);
                    }
                    if (ImGui::DragFloat("Radius", (float*)&m_List[i]->Object->sphere->radius, 0.01f, 0, FLT_MAX,
                                         "%.2f")) {
                        m_World->Object->bvh_node->Destroy();
                        m_World->Object->bvh_node = new (m_World->Object->bvh_node) BVHNode(m_List, 0, m_ListSize);
                    }
                    MaterialNode(m_List[i]->Object->sphere->mat_ptr, i);
                    break;
                case HittableType::XYRECT:
                    if (ImGui::DragFloat3("Position", (float*)&m_List[i]->Object->xy_rect->center, 0.01f, -FLT_MAX,
                                          FLT_MAX, "%.2f")) {
                        m_World->Object->bvh_node->Destroy();
                        m_World->Object->bvh_node = new (m_World->Object->bvh_node) BVHNode(m_List, 0, m_ListSize);
                    }
                    if (ImGui::DragFloat("Width", (float*)&m_List[i]->Object->xy_rect->width, 0.01f, 0, FLT_MAX,
                                         "%.2f")) {
                        m_World->Object->bvh_node->Destroy();
                        m_World->Object->bvh_node = new (m_World->Object->bvh_node) BVHNode(m_List, 0, m_ListSize);
                    }
                    if (ImGui::DragFloat("Height", (float*)&m_List[i]->Object->xy_rect->height, 0.01f, 0, FLT_MAX,
                                         "%.2f")) {
                        m_World->Object->bvh_node->Destroy();
                        m_World->Object->bvh_node = new (m_World->Object->bvh_node) BVHNode(m_List, 0, m_ListSize);
                    }
                    MaterialNode(m_List[i]->Object->xy_rect->mat_ptr, i);
                    break;
                case HittableType::XZRECT:
                    if (ImGui::DragFloat3("Position", (float*)&m_List[i]->Object->xz_rect->center, 0.01f, -FLT_MAX,
                                          FLT_MAX, "%.2f")) {
                        m_World->Object->bvh_node->Destroy();
                        m_World->Object->bvh_node = new (m_World->Object->bvh_node) BVHNode(m_List, 0, m_ListSize);
                    }
                    if (ImGui::DragFloat("Width", (float*)&m_List[i]->Object->xz_rect->width, 0.01f, 0, FLT_MAX,
                                         "%.2f")) {
                        m_World->Object->bvh_node->Destroy();
                        m_World->Object->bvh_node = new (m_World->Object->bvh_node) BVHNode(m_List, 0, m_ListSize);
                    }
                    if (ImGui::DragFloat("Height", (float*)&m_List[i]->Object->xz_rect->height, 0.01f, 0, FLT_MAX,
                                         "%.2f")) {
                        m_World->Object->bvh_node->Destroy();
                        m_World->Object->bvh_node = new (m_World->Object->bvh_node) BVHNode(m_List, 0, m_ListSize);
                    }
                    MaterialNode(m_List[i]->Object->xz_rect->mat_ptr, i);
                    break;
                case HittableType::YZRECT:
                    if (ImGui::DragFloat3("Position", (float*)&m_List[i]->Object->yz_rect->center, 0.01f, -FLT_MAX,
                                          FLT_MAX, "%.2f")) {
                        m_World->Object->bvh_node->Destroy();
                        m_World->Object->bvh_node = new (m_World->Object->bvh_node) BVHNode(m_List, 0, m_ListSize);
                    }
                    if (ImGui::DragFloat("Width", (float*)&m_List[i]->Object->yz_rect->width, 0.01f, 0, FLT_MAX,
                                         "%.2f")) {
                        m_World->Object->bvh_node->Destroy();
                        m_World->Object->bvh_node = new (m_World->Object->bvh_node) BVHNode(m_List, 0, m_ListSize);
                    }
                    if (ImGui::DragFloat("Height", (float*)&m_List[i]->Object->yz_rect->height, 0.01f, 0, FLT_MAX,
                                         "%.2f")) {
                        m_World->Object->bvh_node->Destroy();
                        m_World->Object->bvh_node = new (m_World->Object->bvh_node) BVHNode(m_List, 0, m_ListSize);
                    }
                    MaterialNode(m_List[i]->Object->yz_rect->mat_ptr, i);
                    break;
                default:
                    break;
                }
                ImGui::TreePop();
            }
        }
    }

    ImGui::Separator();

    if (ImGui::Button("Add Hittable...")) {
        ImGui::OpenPopup("New Hittable");
    }

    // Always center this window when appearing
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("New Hittable", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {

        ImGui::Separator();
        ImGui::Text("Choose the type of hittable:");
        ImGui::Separator();

        if (ImGui::Checkbox("Sphere", &m_UseHittableSphere)) {
            m_UseHittableXYRect = false;
            m_UseHittableXZRect = false;
            m_UseHittableYZRect = false;
        }
        else if (ImGui::Checkbox("XYRect", &m_UseHittableXYRect)) {
            m_UseHittableSphere = false;
            m_UseHittableXZRect = false;
            m_UseHittableYZRect = false;
        }
        else if (ImGui::Checkbox("XZRect", &m_UseHittableXZRect)) {
            m_UseHittableSphere = false;
            m_UseHittableXYRect = false;
            m_UseHittableYZRect = false;
        }
        else if (ImGui::Checkbox("YZRect", &m_UseHittableYZRect)) {
            m_UseHittableSphere = false;
            m_UseHittableXYRect = false;
            m_UseHittableXZRect = false;
        }

        ImGui::Separator();
        ImGui::Text("Choose the hittable material:");
        ImGui::Separator();

        if (ImGui::Checkbox("Lambertian", &m_UseLambertian)) {
            m_UseMetal = false;
            m_UseDielectric = false;
            m_UseDiffuseLight = false;
        }
        else if (ImGui::Checkbox("Metal", &m_UseMetal)) {
            m_UseLambertian = false;
            m_UseDielectric = false;
            m_UseDiffuseLight = false;
        }
        else if (ImGui::Checkbox("Dielectric", &m_UseDielectric)) {
            m_UseMetal = false;
            m_UseLambertian = false;
            m_UseDiffuseLight = false;
        }
        else if (ImGui::Checkbox("Diffuse Light", &m_UseDiffuseLight)) {
            m_UseMetal = false;
            m_UseLambertian = false;
            m_UseDielectric = false;
        }

        ImGui::Separator();

        if (m_UseLambertian == true || m_UseMetal == true || m_UseDiffuseLight == true) {
            ImGui::Text("Choose the hittable material texture:");
            ImGui::Separator();
            if (ImGui::Checkbox("Constant Texture", &m_UseConstantTexture)) {
                m_UseCheckerTexture = false;
                m_UseImageTexture = false;
            }
            else if (ImGui::Checkbox("Checker Texture", &m_UseCheckerTexture)) {
                m_UseConstantTexture = false;
                m_UseImageTexture = false;
            }
            else if (ImGui::Checkbox("Image Texture", &m_UseImageTexture)) {
                m_UseConstantTexture = false;
                m_UseCheckerTexture = false;
            }
            ImGui::Separator();
        }

        if (((m_UseLambertian || m_UseMetal || m_UseDielectric || m_UseDiffuseLight) &&
             (m_UseHittableSphere || m_UseHittableXYRect || m_UseHittableXZRect || m_UseHittableYZRect))) {
            if (!m_UseDielectric) {
                if (m_UseConstantTexture || m_UseCheckerTexture || m_UseImageTexture) {
                    if (ImGui::Button("Add")) {
                        AddHittable();
                        ImGui::CloseCurrentPopup();
                    }
                }
            }
            else {
                if (ImGui::Button("Add")) {
                    AddHittable();
                    ImGui::CloseCurrentPopup();
                }
            }
        }

        if (ImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    if (ImGui::Button("Delete Hittable...")) {
        ImGui::OpenPopup("Delete Hittable");
    }

    // Always center this window when appearing
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if (ImGui::BeginPopupModal("Delete Hittable", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Enter the hittable ID you want to delete");
        ImGui::InputInt("Hittable ID", &m_HittableID);

        for (int i = 0; i < m_ListSize; i++) {
            if (m_HittableID == i) {
                if (ImGui::Button("Delete")) {
                    DeleteHittable(m_List[i], i);
                    ImGui::CloseCurrentPopup();
                }
            }
        }

        if (ImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    ImGui::End();

    ImGui::Begin("Opions");

    ImGui::Separator();

    if (ImGui::CollapsingHeader("Camera Settings", base_flags)) {
        if (ImGui::DragFloat3("Position", (float*)&m_Camera->m_Position, 0.01f, -FLT_MAX, FLT_MAX, "%.2f")) {
            m_Inputs.origin_x = m_Camera->m_Position.x;
            m_Inputs.origin_y = m_Camera->m_Position.y;
            m_Inputs.origin_z = m_Camera->m_Position.z;
        }
        if (ImGui::DragFloat3("Orientation", (float*)&m_Camera->m_Orientation, 0.01f, -FLT_MAX, FLT_MAX, "%.2f")) {
            m_Inputs.orientation_x = m_Camera->m_Orientation.x;
            m_Inputs.orientation_y = m_Camera->m_Orientation.y;
            m_Inputs.orientation_z = m_Camera->m_Orientation.z;
        }
        if (ImGui::SliderFloat("Field of view", &m_Camera->m_Fov, 1.0f, 120.0f, "%.f")) {
            m_Inputs.fov = glm::radians(m_Camera->m_Fov);
        }
    }

    ImGui::Separator();

    if (ImGui::CollapsingHeader("Ray Tracing Settings", base_flags)) {
        ImGui::InputInt("Samples Per Pixel", (int*)&m_SamplesPerPixel, 1, 100);
        ImGui::SliderInt("Max Depth", (int*)&m_MaxDepth, 1, 50);
    }

    ImGui::Separator();

    ImGui::End();
}

bool CudaLayer::OnImGuiResize()
{
    ImVec2 view = ImGui::GetContentRegionAvail();

    if (view.x != m_ImageWidth || view.y != m_ImageHeight) {
        if (view.x == 0 || view.y == 0) {
            return false;
        }

        m_ImageWidth = view.x;
        m_ImageHeight = view.y;

        m_NumTexels = m_ImageWidth * m_ImageHeight;
        m_NumValues = m_NumTexels * 4;
        m_SizeTexData = sizeof(GLubyte) * m_NumValues;

        checkCudaErrors(cudaFree(m_CudaDevRenderBuffer));
        checkCudaErrors(cudaFree(m_DrandState));
        checkCudaErrors(cudaFree(m_DrandState2));
        checkCudaErrors(cudaGraphicsUnregisterResource(m_CudaTexResource));

        InitCudaBuffers();
        InitGLBuffers();
        RunCudaInit();

        return true;
    }
    return true;
}

void CudaLayer::MaterialNode(Material* material, size_t i)
{
    if (ImGui::TreeNodeEx("Material", base_flags)) {
        const char* mat_items[] = {"Lambertian", "Metal", "Dielectric", "Diffuse Light"};
        int mat_item_current = material->type;

        if (ImGui::Combo(" ", &mat_item_current, mat_items, IM_ARRAYSIZE(mat_items))) {
            material->type = (MaterialType)mat_item_current;
            switch (material->type) {
            case MaterialType::LAMBERTIAN:
                material->Object->lambertian = (Lambertian*)(material->Object + 1);
                material->Object->lambertian->albedo = (Texture*)(material->Object->lambertian + 1);
                material->Object->lambertian->albedo->Object =
                    (Texture::ObjectUnion*)(material->Object->lambertian->albedo + 1);
                material->Object->lambertian->albedo->Object->checker =
                    (Checker*)(material->Object->lambertian->albedo->Object + 1);
                material->Object->lambertian->albedo->Object->checker->odd =
                    (Constant*)(material->Object->lambertian->albedo->Object->checker + 1);
                material->Object->lambertian->albedo->Object->checker->even =
                    (Constant*)(material->Object->lambertian->albedo->Object->checker->odd + 1);
                new (material->Object->lambertian->albedo->Object->constant) Constant(Vec3(0.9f, 0.9f, 0.9f));
                material->Object->lambertian->albedo->type = TextureType::CONSTANT;
                new (material->Object->lambertian) Lambertian(material->Object->lambertian->albedo);
                break;
            case MaterialType::METAL:
                material->Object->metal = (Metal*)(material->Object + 1);
                material->Object->metal->albedo = (Texture*)(material->Object->metal + 1);
                material->Object->metal->albedo->Object = (Texture::ObjectUnion*)(material->Object->metal->albedo + 1);
                material->Object->metal->albedo->Object->checker =
                    (Checker*)(material->Object->metal->albedo->Object + 1);
                material->Object->metal->albedo->Object->checker->odd =
                    (Constant*)(material->Object->metal->albedo->Object->checker + 1);
                material->Object->metal->albedo->Object->checker->even =
                    (Constant*)(material->Object->metal->albedo->Object->checker->odd + 1);
                new (material->Object->metal->albedo->Object->constant) Constant(Vec3(0.9f, 0.9f, 0.9f));
                material->Object->metal->albedo->type = TextureType::CONSTANT;
                new (material->Object->metal) Metal(material->Object->metal->albedo, 0.0f);
                break;
            case MaterialType::DIELECTRIC:
                material->Object->dielectric = (Dielectric*)(material->Object + 1);
                new (material->Object->dielectric) Dielectric(0.0f);
                break;
            case MaterialType::DIFFUSELIGHT:
                material->Object->diffuse_light = (DiffuseLight*)(material->Object + 1);
                material->Object->diffuse_light->albedo = (Texture*)(material->Object->diffuse_light + 1);
                material->Object->diffuse_light->albedo->Object =
                    (Texture::ObjectUnion*)(material->Object->diffuse_light->albedo + 1);
                material->Object->diffuse_light->albedo->Object->checker =
                    (Checker*)(material->Object->diffuse_light->albedo->Object + 1);
                material->Object->diffuse_light->albedo->Object->checker->odd =
                    (Constant*)(material->Object->diffuse_light->albedo->Object->checker + 1);
                material->Object->diffuse_light->albedo->Object->checker->even =
                    (Constant*)(material->Object->diffuse_light->albedo->Object->checker->odd + 1);
                new (material->Object->diffuse_light->albedo->Object->constant) Constant(Vec3(0.9f, 0.9f, 0.9f));
                material->Object->diffuse_light->albedo->type = TextureType::CONSTANT;
                new (material->Object->diffuse_light) DiffuseLight(material->Object->diffuse_light->albedo, 3);
                break;
            default:
                break;
            }
        }

        // Material switch-case
        switch (material->type) {
        case MaterialType::LAMBERTIAN:
            TextureNode(material->Object->lambertian->albedo, i);
            break;
        case MaterialType::METAL:
            ImGui::DragFloat("Fuzziness", (float*)&material->Object->metal->fuzz, 0.01f, 0.0f, 1.0f, "%.2f");
            TextureNode(material->Object->metal->albedo, i);
            break;
        case MaterialType::DIELECTRIC:
            ImGui::DragFloat("Index of Refraction", (float*)&material->Object->dielectric->ir, 0.01f, 0.0f, FLT_MAX,
                             "%.2f");
            break;
        case MaterialType::DIFFUSELIGHT:
            ImGui::SliderInt("Light Intensity", &material->Object->diffuse_light->light_intensity, 0, 10);
            TextureNode(material->Object->diffuse_light->albedo, i);
            break;
        default:
            break;
        }

        ImGui::TreePop();
    }
}

void CudaLayer::TextureNode(Texture* texture, size_t i)
{
    if (ImGui::TreeNodeEx("Texture", base_flags)) {
        const char* tex_items[] = {"Constant", "Checker", "Image"};
        int tex_item_current = texture->type;

        if (ImGui::Combo(" ", &tex_item_current, tex_items, IM_ARRAYSIZE(tex_items))) {
            texture->type = (TextureType)tex_item_current;
            switch (texture->type) {
            case TextureType::CONSTANT:
                texture->Object->constant = (Constant*)(texture->Object + 1);
                new (texture->Object->constant) Constant(Vec3(0.9f, 0.9f, 0.9f));
                break;
            case TextureType::CHECKER:
                texture->Object->checker = (Checker*)(texture->Object + 1);
                texture->Object->checker->odd = (Constant*)(texture->Object->checker + 1);
                texture->Object->checker->even = (Constant*)(texture->Object->checker->odd + 1);
                new (texture->Object->checker->odd) Constant(Vec3(0.2f, 0.3f, 0.1f));
                new (texture->Object->checker->even) Constant(Vec3(0.9f, 0.9f, 0.9f));
                new (texture->Object->checker) Checker(texture->Object->checker->odd, texture->Object->checker->even);
                break;
            case TextureType::IMAGE:
                texture->Object->image->data = nullptr;
                texture->Object->image->path = nullptr;
                break;
            default:
                break;
            }
        }

        // Texture switch-case
        switch (texture->type) {
        case TextureType::CONSTANT:
            ImGui::ColorEdit3("Albedo", (float*)&texture->Object->constant->color);
            break;
        case TextureType::CHECKER:
            ImGui::ColorEdit3("Albedo odd", (float*)&texture->Object->checker->odd->color);
            ImGui::ColorEdit3("Albedo even", (float*)&texture->Object->checker->even->color);
            break;
        case TextureType::IMAGE:
            if (ImGui::Button("Open...")) {
                m_ButtonID = i;
                // clang-format off
                ImGuiFileDialog::Instance()->OpenDialog(
                    "ChooseFileDlgKey",
                    "Choose File ",
                    ".jpg,.jpeg,.png",
                    ".",
                    1, nullptr, ImGuiFileDialogFlags_Modal);
                // clang-format on
            }
            if (m_ButtonID == i) {
                ImageAllocation(texture->Object->image);
            }
            if (texture->Object->image->path != nullptr) {
                ImGui::Text("%s", texture->Object->image->path);
            }
            else {
                ImGui::Text("None");
            }
            break;
        default:
            break;
        }
        ImGui::TreePop();
    }
}

void CudaLayer::ImageAllocation(Image* image)
{
    // Always center this window when appearing
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    // display
    if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey")) {
        // action if OK
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string temp = ("assets/textures/" + ImGuiFileDialog::Instance()->GetCurrentFileName());
            m_TextureImageFilename = (char*)malloc(temp.length() + 1);
            std::strcpy(m_TextureImageFilename, temp.c_str());

            if (image->data != nullptr) {
                checkCudaErrors(cudaFree(image->data));
            }

            if (image->path != nullptr) {
                checkCudaErrors(cudaFree((void*)image->path));
            }

            m_TextureImageData = LoadImage(m_TextureImageFilename, m_TextureImageData, &m_TextureImageWidth,
                                           &m_TextureImageHeight, &m_TextureImageNR);
            checkCudaErrors(cudaMallocManaged(&image->data, m_TextureImageWidth * m_TextureImageHeight *
                                                                m_TextureImageNR * sizeof(unsigned char)));
            checkCudaErrors(
                cudaMemcpy(image->data, m_TextureImageData,
                           m_TextureImageWidth * m_TextureImageHeight * m_TextureImageNR * sizeof(unsigned char),
                           cudaMemcpyHostToDevice));
            STBI_FREE(m_TextureImageData);
            checkCudaErrors(cudaMallocManaged(&image->path, 64 * sizeof(const char)));
            checkCudaErrors(cudaMemcpy((void*)image->path, m_TextureImageFilename, 64 * sizeof(const char),
                                       cudaMemcpyHostToDevice));

            new (image) Image(image->data, image->path, m_TextureImageWidth, m_TextureImageHeight);
            free(m_TextureImageFilename);
        }

        // close
        ImGuiFileDialog::Instance()->Close();
    }
}

void CudaLayer::AddHittable()
{
    Hittable* hittable;
    int i;
    bool allocateNew = false;

    if (!m_InactiveHittables.empty()) {
        for (auto& inactiveHittable : m_InactiveHittables) {
            hittable = inactiveHittable.first;
            i = inactiveHittable.second;
            if (m_UseHittableSphere) {
                if (hittable->type == HittableType::SPHERE) {
                    // Reuse an inactive Hittable
                    hittable->isActive = true;

                    new (hittable->Object->sphere->mat_ptr->Object->lambertian->albedo->Object->constant)
                        Constant(Vec3(0.9f, 0.9f, 0.9f));
                    hittable->Object->sphere->mat_ptr->Object->lambertian->albedo->type = TextureType::CONSTANT;
                    new (hittable->Object->sphere->mat_ptr->Object->lambertian)
                        Lambertian(hittable->Object->sphere->mat_ptr->Object->lambertian->albedo);
                    // Set the type of the Material after constructing it, so the assignment won't be overwritten.
                    hittable->Object->sphere->mat_ptr->type = MaterialType::LAMBERTIAN;
                    hittable->Object->sphere = new (hittable->Object->sphere)
                        Sphere(Vec3(0.0f, 0.0f, 0.0f), 1.0f, hittable->Object->sphere->mat_ptr);

                    m_List[i] = hittable;

                    m_World->Object->bvh_node->Destroy();
                    m_World->Object->bvh_node = new (m_World->Object->bvh_node) BVHNode(m_List, 0, m_ListSize);

                    allocateNew = false;
                    break;
                }
                else {
                    allocateNew = true;
                }
            }
            else if (m_UseHittableXYRect) {
                if (hittable->type == HittableType::XYRECT) {
                    // Reuse an inactive Hittable
                    hittable->isActive = true;

                    new (hittable->Object->xy_rect->mat_ptr->Object->lambertian->albedo->Object->constant)
                        Constant(Vec3(0.9f, 0.9f, 0.9f));
                    hittable->Object->xy_rect->mat_ptr->Object->lambertian->albedo->type = TextureType::CONSTANT;
                    new (hittable->Object->xy_rect->mat_ptr->Object->lambertian)
                        Lambertian(hittable->Object->xy_rect->mat_ptr->Object->lambertian->albedo);
                    // Set the type of the Material after constructing it, so the assignment won't be overwritten.
                    hittable->Object->xy_rect->mat_ptr->type = MaterialType::LAMBERTIAN;
                    hittable->Object->xy_rect = new (hittable->Object->xy_rect)
                        XYRect(Vec3(0.0f, 0.0f, 0.0f), 1.0f, 1.0f, hittable->Object->xy_rect->mat_ptr);

                    m_List[i] = hittable;

                    m_World->Object->bvh_node->Destroy();
                    m_World->Object->bvh_node = new (m_World->Object->bvh_node) BVHNode(m_List, 0, m_ListSize);

                    allocateNew = false;
                    break;
                }
                else {
                    allocateNew = true;
                }
            }
            else if (m_UseHittableXZRect) {
                if (hittable->type == HittableType::XZRECT) {
                    // Reuse an inactive Hittable
                    hittable->isActive = true;

                    new (hittable->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->constant)
                        Constant(Vec3(0.9f, 0.9f, 0.9f));
                    hittable->Object->xz_rect->mat_ptr->Object->lambertian->albedo->type = TextureType::CONSTANT;
                    new (hittable->Object->xz_rect->mat_ptr->Object->lambertian)
                        Lambertian(hittable->Object->xz_rect->mat_ptr->Object->lambertian->albedo);
                    // Set the type of the Material after constructing it, so the assignment won't be overwritten.
                    hittable->Object->xz_rect->mat_ptr->type = MaterialType::LAMBERTIAN;
                    hittable->Object->xz_rect = new (hittable->Object->xz_rect)
                        XZRect(Vec3(0.0f, 0.0f, 0.0f), 1.0f, 1.0f, hittable->Object->xz_rect->mat_ptr);

                    m_List[i] = hittable;

                    m_World->Object->bvh_node->Destroy();
                    m_World->Object->bvh_node = new (m_World->Object->bvh_node) BVHNode(m_List, 0, m_ListSize);

                    allocateNew = false;
                    break;
                }
                else {
                    allocateNew = true;
                }
            }
            else if (m_UseHittableYZRect) {
                if (hittable->type == HittableType::YZRECT) {
                    // Reuse an inactive Hittable
                    hittable->isActive = true;

                    new (hittable->Object->yz_rect->mat_ptr->Object->lambertian->albedo->Object->constant)
                        Constant(Vec3(0.9f, 0.9f, 0.9f));
                    hittable->Object->yz_rect->mat_ptr->Object->lambertian->albedo->type = TextureType::CONSTANT;
                    new (hittable->Object->yz_rect->mat_ptr->Object->lambertian)
                        Lambertian(hittable->Object->yz_rect->mat_ptr->Object->lambertian->albedo);
                    // Set the type of the Material after constructing it, so the assignment won't be overwritten.
                    hittable->Object->yz_rect->mat_ptr->type = MaterialType::LAMBERTIAN;
                    hittable->Object->yz_rect = new (hittable->Object->yz_rect)
                        YZRect(Vec3(0.0f, 0.0f, 0.0f), 1.0f, 1.0f, hittable->Object->yz_rect->mat_ptr);

                    m_List[i] = hittable;

                    m_World->Object->bvh_node->Destroy();
                    m_World->Object->bvh_node = new (m_World->Object->bvh_node) BVHNode(m_List, 0, m_ListSize);

                    allocateNew = false;
                    break;
                }
                else {
                    allocateNew = true;
                }
            }
        }
    }
    else {
        allocateNew = true;
    }

    if (allocateNew) {
        std::cout << "Allocate NEW Hittable" << std::endl;
        if (m_UseHittableSphere) {
            size_t newSphere = m_SphereSize + m_MetalSize + m_CheckerSize;

            m_ListSize++;
            m_TotalSize += newSphere;

            if (temp != nullptr) {
                cudaFree(temp);
            }
            checkCudaErrors(cudaMallocManaged(&temp, m_TotalSize - newSphere));
            checkCudaErrors(cudaMemcpy(temp, m_ListMemory, m_TotalSize - newSphere, cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaFree(m_ListMemory));
            checkCudaErrors(cudaMallocManaged(&m_ListMemory, m_TotalSize));
            m_ListMemory = temp;

            // Reallocate the list memory
            // char* newListMemory;
            // checkCudaErrors(cudaMallocManaged(&newListMemory, m_TotalSize - newSphere));
            // checkCudaErrors(cudaMemcpy(newListMemory, m_ListMemory, m_TotalSize - newSphere,
            // cudaMemcpyDeviceToDevice)); checkCudaErrors(cudaFree(m_ListMemory));
            // checkCudaErrors(cudaMallocManaged(&m_ListMemory, m_TotalSize));
            // checkCudaErrors(cudaMemcpy(m_ListMemory, newListMemory, m_TotalSize - newSphere,
            // cudaMemcpyDeviceToDevice)); checkCudaErrors(cudaFree(newListMemory));

            // Update the list pointer
            m_List = (Hittable**)m_ListMemory;

            // Partitioning
            char* basePtr = m_ListMemory + (m_TotalSize - newSphere);
            m_List[m_ListSize - 1] = (Hittable*)(basePtr);
            m_List[m_ListSize - 1]->Object = (Hittable::ObjectUnion*)(m_List[m_ListSize - 1] + 1);
            m_List[m_ListSize - 1]->Object->sphere = (Sphere*)(m_List[m_ListSize - 1]->Object + 1);
            m_List[m_ListSize - 1]->Object->sphere->mat_ptr = (Material*)(m_List[m_ListSize - 1]->Object->sphere + 1);
            m_List[m_ListSize - 1]->Object->sphere->mat_ptr->Object =
                (Material::ObjectUnion*)(m_List[m_ListSize - 1]->Object->sphere->mat_ptr + 1);
            m_List[m_ListSize - 1]->Object->sphere->mat_ptr->Object->metal =
                (Metal*)(m_List[m_ListSize - 1]->Object->sphere->mat_ptr->Object + 1);
            m_List[m_ListSize - 1]->Object->sphere->mat_ptr->Object->metal->albedo =
                (Texture*)(m_List[m_ListSize - 1]->Object->sphere->mat_ptr->Object->metal + 1);
            m_List[m_ListSize - 1]->Object->sphere->mat_ptr->Object->metal->albedo->Object =
                (Texture::ObjectUnion*)(m_List[m_ListSize - 1]->Object->sphere->mat_ptr->Object->metal->albedo + 1);
            m_List[m_ListSize - 1]->Object->sphere->mat_ptr->Object->metal->albedo->Object->checker =
                (Checker*)(m_List[m_ListSize - 1]->Object->sphere->mat_ptr->Object->metal->albedo->Object + 1);
            m_List[m_ListSize - 1]->Object->sphere->mat_ptr->Object->metal->albedo->Object->checker->odd =
                (Constant*)(m_List[m_ListSize - 1]->Object->sphere->mat_ptr->Object->metal->albedo->Object->checker +
                            1);
            m_List[m_ListSize - 1]->Object->sphere->mat_ptr->Object->metal->albedo->Object->checker->even =
                (Constant*)(m_List[m_ListSize - 1]
                                ->Object->sphere->mat_ptr->Object->metal->albedo->Object->checker->odd +
                            1);

            m_List[m_ListSize - 1]->type = HittableType::SPHERE;
            m_List[m_ListSize - 1]->isActive = true;

            new (m_List[m_ListSize - 1]->Object->sphere->mat_ptr->Object->lambertian->albedo->Object->constant)
                Constant(Vec3(0.9f, 0.9f, 0.9f));
            m_List[m_ListSize - 1]->Object->sphere->mat_ptr->Object->lambertian->albedo->type = TextureType::CONSTANT;
            new (m_List[m_ListSize - 1]->Object->sphere->mat_ptr->Object->lambertian)
                Lambertian(m_List[m_ListSize - 1]->Object->sphere->mat_ptr->Object->lambertian->albedo);
            // Set the type of the Material after constructing it, so the assignment won't be overwritten.
            m_List[m_ListSize - 1]->Object->sphere->mat_ptr->type = MaterialType::LAMBERTIAN;
            m_List[m_ListSize - 1]->Object->sphere = new (m_List[m_ListSize - 1]->Object->sphere)
                Sphere(Vec3(0.0f, 0.0f, 0.0f), 1.0f, m_List[m_ListSize - 1]->Object->sphere->mat_ptr);

            m_World->Object->bvh_node->Destroy();
            m_World->Object->bvh_node = new (m_World->Object->bvh_node) BVHNode(m_List, 0, m_ListSize);
        }
        else if (m_UseHittableXYRect) {
        }
        else if (m_UseHittableXZRect) {
        }
        else if (m_UseHittableYZRect) {
        }
    }
    else {
        m_InactiveHittables.pop_back();
    }
}

void CudaLayer::DeleteHittable(Hittable* hittable, int i)
{
    hittable->isActive = false;
    m_World->Object->bvh_node->Destroy();
    DeleteImageAllocation(hittable);
    m_World->Object->bvh_node = new (m_World->Object->bvh_node) BVHNode(m_List, 0, m_ListSize);
    m_InactiveHittables.push_back(std::make_pair(hittable, i));

    for (size_t i = 0; i < m_ListSize; i++) {
        printf("i = %zu\n", i);
        std::cout << "Hexadecimal representation of pointer: " << std::hex
                  << reinterpret_cast<unsigned long long>(m_List[i]) << std::endl;
        std::cout << "Decimal representation of pointer: " << std::dec
                  << reinterpret_cast<unsigned long long>(m_List[i]) << std::endl;
    }
}

void CudaLayer::DeleteImageAllocation(Hittable* hittable)
{
    switch (hittable->type) {
    case HittableType::SPHERE:
        switch (hittable->Object->sphere->mat_ptr->type) {
        case MaterialType::LAMBERTIAN:
            if (hittable->Object->sphere->mat_ptr->Object->lambertian->albedo->type == TextureType::IMAGE) {
                if (hittable->Object->sphere->mat_ptr->Object->lambertian->albedo->Object->image->data != nullptr) {
                    checkCudaErrors(
                        cudaFree(hittable->Object->sphere->mat_ptr->Object->lambertian->albedo->Object->image->data));
                }
                if (hittable->Object->sphere->mat_ptr->Object->lambertian->albedo->Object->image->path != nullptr) {
                    checkCudaErrors(cudaFree(
                        (void*)hittable->Object->sphere->mat_ptr->Object->lambertian->albedo->Object->image->path));
                }
            }
            break;
        case MaterialType::METAL:
            if (hittable->Object->sphere->mat_ptr->Object->metal->albedo->type == TextureType::IMAGE) {
                if (hittable->Object->sphere->mat_ptr->Object->metal->albedo->Object->image->data != nullptr) {
                    checkCudaErrors(
                        cudaFree(hittable->Object->sphere->mat_ptr->Object->metal->albedo->Object->image->data));
                }
                if (hittable->Object->sphere->mat_ptr->Object->metal->albedo->Object->image->path != nullptr) {
                    checkCudaErrors(
                        cudaFree((void*)hittable->Object->sphere->mat_ptr->Object->metal->albedo->Object->image->path));
                }
            }
            break;
        case MaterialType::DIFFUSELIGHT:
            if (hittable->Object->sphere->mat_ptr->Object->diffuse_light->albedo->type == TextureType::IMAGE) {
                if (hittable->Object->sphere->mat_ptr->Object->diffuse_light->albedo->Object->image->data != nullptr) {
                    checkCudaErrors(cudaFree(
                        hittable->Object->sphere->mat_ptr->Object->diffuse_light->albedo->Object->image->data));
                }
                if (hittable->Object->sphere->mat_ptr->Object->diffuse_light->albedo->Object->image->path != nullptr) {
                    checkCudaErrors(cudaFree(
                        (void*)hittable->Object->sphere->mat_ptr->Object->diffuse_light->albedo->Object->image->path));
                }
            }
            break;
        default:
            break;
        }
        break;
    case HittableType::XYRECT:
        switch (hittable->Object->xy_rect->mat_ptr->type) {
        case MaterialType::LAMBERTIAN:
            if (hittable->Object->xy_rect->mat_ptr->Object->lambertian->albedo->type == TextureType::IMAGE) {
                if (hittable->Object->xy_rect->mat_ptr->Object->lambertian->albedo->Object->image->data != nullptr) {
                    checkCudaErrors(
                        cudaFree(hittable->Object->xy_rect->mat_ptr->Object->lambertian->albedo->Object->image->data));
                }
                if (hittable->Object->xy_rect->mat_ptr->Object->lambertian->albedo->Object->image->path != nullptr) {
                    checkCudaErrors(cudaFree(
                        (void*)hittable->Object->xy_rect->mat_ptr->Object->lambertian->albedo->Object->image->path));
                }
            }
            break;
        case MaterialType::METAL:
            if (hittable->Object->xy_rect->mat_ptr->Object->metal->albedo->type == TextureType::IMAGE) {
                if (hittable->Object->xy_rect->mat_ptr->Object->metal->albedo->Object->image->data != nullptr) {
                    checkCudaErrors(
                        cudaFree(hittable->Object->xy_rect->mat_ptr->Object->metal->albedo->Object->image->data));
                }
                if (hittable->Object->xy_rect->mat_ptr->Object->metal->albedo->Object->image->path != nullptr) {
                    checkCudaErrors(cudaFree(
                        (void*)hittable->Object->xy_rect->mat_ptr->Object->metal->albedo->Object->image->path));
                }
            }
            break;
        case MaterialType::DIFFUSELIGHT:
            if (hittable->Object->xy_rect->mat_ptr->Object->diffuse_light->albedo->type == TextureType::IMAGE) {
                if (hittable->Object->xy_rect->mat_ptr->Object->diffuse_light->albedo->Object->image->data != nullptr) {
                    checkCudaErrors(cudaFree(
                        hittable->Object->xy_rect->mat_ptr->Object->diffuse_light->albedo->Object->image->data));
                }
                if (hittable->Object->xy_rect->mat_ptr->Object->diffuse_light->albedo->Object->image->path != nullptr) {
                    checkCudaErrors(cudaFree(
                        (void*)hittable->Object->xy_rect->mat_ptr->Object->diffuse_light->albedo->Object->image->path));
                }
            }
            break;
        default:
            break;
        }
        break;
    case HittableType::XZRECT:
        switch (hittable->Object->xz_rect->mat_ptr->type) {
        case MaterialType::LAMBERTIAN:
            if (hittable->Object->xz_rect->mat_ptr->Object->lambertian->albedo->type == TextureType::IMAGE) {
                if (hittable->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->image->data != nullptr) {
                    checkCudaErrors(
                        cudaFree(hittable->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->image->data));
                }
                if (hittable->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->image->path != nullptr) {
                    checkCudaErrors(cudaFree(
                        (void*)hittable->Object->xz_rect->mat_ptr->Object->lambertian->albedo->Object->image->path));
                }
            }
            break;
        case MaterialType::METAL:
            if (hittable->Object->xz_rect->mat_ptr->Object->metal->albedo->type == TextureType::IMAGE) {
                if (hittable->Object->xz_rect->mat_ptr->Object->metal->albedo->Object->image->data != nullptr) {
                    checkCudaErrors(
                        cudaFree(hittable->Object->xz_rect->mat_ptr->Object->metal->albedo->Object->image->data));
                }
                if (hittable->Object->xz_rect->mat_ptr->Object->metal->albedo->Object->image->path != nullptr) {
                    checkCudaErrors(cudaFree(
                        (void*)hittable->Object->xz_rect->mat_ptr->Object->metal->albedo->Object->image->path));
                }
            }
            break;
        case MaterialType::DIFFUSELIGHT:
            if (hittable->Object->xz_rect->mat_ptr->Object->diffuse_light->albedo->type == TextureType::IMAGE) {
                if (hittable->Object->xz_rect->mat_ptr->Object->diffuse_light->albedo->Object->image->data != nullptr) {
                    checkCudaErrors(cudaFree(
                        hittable->Object->xz_rect->mat_ptr->Object->diffuse_light->albedo->Object->image->data));
                }
                if (hittable->Object->xz_rect->mat_ptr->Object->diffuse_light->albedo->Object->image->path != nullptr) {
                    checkCudaErrors(cudaFree(
                        (void*)hittable->Object->xz_rect->mat_ptr->Object->diffuse_light->albedo->Object->image->path));
                }
            }
            break;
        default:
            break;
        }
        break;
    case HittableType::YZRECT:
        switch (hittable->Object->yz_rect->mat_ptr->type) {
        case MaterialType::LAMBERTIAN:
            if (hittable->Object->yz_rect->mat_ptr->Object->lambertian->albedo->type == TextureType::IMAGE) {
                if (hittable->Object->yz_rect->mat_ptr->Object->lambertian->albedo->Object->image->data != nullptr) {
                    checkCudaErrors(
                        cudaFree(hittable->Object->yz_rect->mat_ptr->Object->lambertian->albedo->Object->image->data));
                }
                if (hittable->Object->yz_rect->mat_ptr->Object->lambertian->albedo->Object->image->path != nullptr) {
                    checkCudaErrors(cudaFree(
                        (void*)hittable->Object->yz_rect->mat_ptr->Object->lambertian->albedo->Object->image->path));
                }
            }
            break;
        case MaterialType::METAL:
            if (hittable->Object->yz_rect->mat_ptr->Object->metal->albedo->type == TextureType::IMAGE) {
                if (hittable->Object->yz_rect->mat_ptr->Object->metal->albedo->Object->image->data != nullptr) {
                    checkCudaErrors(
                        cudaFree(hittable->Object->yz_rect->mat_ptr->Object->metal->albedo->Object->image->data));
                }
                if (hittable->Object->yz_rect->mat_ptr->Object->metal->albedo->Object->image->path != nullptr) {
                    checkCudaErrors(cudaFree(
                        (void*)hittable->Object->yz_rect->mat_ptr->Object->metal->albedo->Object->image->path));
                }
            }
            break;
        case MaterialType::DIFFUSELIGHT:
            if (hittable->Object->yz_rect->mat_ptr->Object->diffuse_light->albedo->type == TextureType::IMAGE) {
                if (hittable->Object->yz_rect->mat_ptr->Object->diffuse_light->albedo->Object->image->data != nullptr) {
                    checkCudaErrors(cudaFree(
                        hittable->Object->yz_rect->mat_ptr->Object->diffuse_light->albedo->Object->image->data));
                }
                if (hittable->Object->yz_rect->mat_ptr->Object->diffuse_light->albedo->Object->image->path != nullptr) {
                    checkCudaErrors(cudaFree(
                        (void*)hittable->Object->yz_rect->mat_ptr->Object->diffuse_light->albedo->Object->image->path));
                }
            }
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }
}

void CudaLayer::OnDetach()
{
    m_World->Object->bvh_node->Destroy();
    for (size_t i = 0; i < m_ListSize; i++) {
        if (m_List[i]->isActive)
            DeleteImageAllocation(m_List[i]);
    }
    checkCudaErrors(cudaFree(m_ListMemory));
    checkCudaErrors(cudaFree(m_WorldMemory));

    checkCudaErrors(cudaFree(m_CudaDevRenderBuffer));
    checkCudaErrors(cudaFree(m_DrandState));
    checkCudaErrors(cudaFree(m_DrandState2));

    checkCudaErrors(cudaGraphicsUnregisterResource(m_CudaTexResource));

    // useful for compute-sanitizer --leak-check full
    cudaDeviceReset();
}
