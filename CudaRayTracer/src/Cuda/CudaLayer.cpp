#include "Cuda/CudaLayer.h"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

extern "C"
void LaunchKernel(unsigned int *pos, unsigned int image_width, unsigned int image_height, const unsigned int samples_per_pixel, const unsigned int max_depth, HittableList* world, curandState *d_rand_state, InputStruct inputs);

extern "C" void LaunchRandInit(curandState *d_rand_state2);

extern "C"
void LaunchRenderInit(dim3 grid, dim3 block, unsigned int image_width, unsigned int image_height, curandState *d_rand_state);

// extern "C"
// void LaunchCreateWorld(Hittable **d_list, Hittable **d_world, const float aspect_ratio, curandState *d_rand_state2);

// extern "C"
// void LaunchFreeWorld(Hittable **d_list, Hittable **d_world, const unsigned int num_hittables);

CudaLayer::CudaLayer()
    : Layer("CudaLayer")
{
    m_World = new HittableList();
}

void CudaLayer::OnAttach()
{
    findCudaDevice();

    InitCudaBuffers();
    InitGLBuffers();

    RunCudaInit();

    GenerateWorld();

    m_Camera = std::make_unique<Camera>(m_ImageWidth, m_ImageHeight, glm::vec3(0.0f, 0.0f, 3.0f));

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
    m_Inputs.fov = m_Camera->m_Fov;
}

void CudaLayer::OnDetach()
{
    checkCudaErrors(cudaFree(m_CudaDevRenderBuffer));
    checkCudaErrors(cudaFree(m_DrandState));
    checkCudaErrors(cudaFree(m_DrandState2));

    for (auto obj : m_World->objects) {
        checkCudaErrors(cudaFree(obj));
    }

    checkCudaErrors(cudaGraphicsUnregisterResource(m_CudaTexResource));

    // useful for compute-sanitizer --leak-check full
    cudaDeviceReset();
}

void CudaLayer::OnUpdate()
{
    RunCudaUpdate();
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

    if (ImGui::IsItemActive()) {

        m_Camera->Inputs((GLFWwindow *)ImGui::GetMainViewport()->PlatformHandle);

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
        glfwSetInputMode((GLFWwindow *)ImGui::GetMainViewport()->PlatformHandle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        m_Camera->m_FirstClick = true;
    }

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
        checkCudaErrors(cudaMalloc(&m_CudaDevRenderBuffer, m_SizeTexData));
        checkCudaErrors(cudaMalloc((void **)&m_DrandState, m_NumTexels * sizeof(curandState)));
        checkCudaErrors(cudaMalloc((void **)&m_DrandState2, 1 * sizeof(curandState)));

        InitGLBuffers();
        RunCudaInit();

        return true;
    }
    return true;
}

void CudaLayer::InitCudaBuffers()
{
    checkCudaErrors(cudaMalloc(&m_CudaDevRenderBuffer, m_SizeTexData)); // Allocate CUDA memory for color output
    // Allocate random state
    checkCudaErrors(cudaMalloc((void **)&m_DrandState, m_NumTexels * sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void **)&m_DrandState2, 1 * sizeof(curandState)));
}

void CudaLayer::InitGLBuffers()
{
    // create an OpenGL texture
    glGenTextures(1, &m_Texture);              // generate 1 texture
    glBindTexture(GL_TEXTURE_2D, m_Texture); // set it as current target
    // set basic texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // Specify 2D texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_ImageWidth, m_ImageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    // Register this texture with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterImage(&m_CudaTexResource, m_Texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    // SDK_CHECK_ERROR_GL();
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
    Sphere* groundSphere;
    checkCudaErrors(cudaMallocManaged(&groundSphere, sizeof(Sphere)));
    checkCudaErrors(cudaMallocManaged(&groundSphere->mat_ptr, sizeof(Material)));
    m_World->Add(new(groundSphere) Sphere(Vec3(0.0f, -100.5f, 0.0f), 100.0f, new(groundSphere->mat_ptr) Material(Vec3(0.8f, 0.8f, 0.0f), Mat::lambertian)));

    Sphere* sphere1;
    checkCudaErrors(cudaMallocManaged(&sphere1, sizeof(Sphere)));
    checkCudaErrors(cudaMallocManaged(&sphere1->mat_ptr, sizeof(Material)));
    m_World->Add(new(sphere1) Sphere(Vec3(0.0f, 0.0f, -1.0f), 0.5f, new(sphere1->mat_ptr) Material(Vec3(0.1f, 0.2f, 0.5f), Mat::lambertian)));

    Sphere* sphere2;
    checkCudaErrors(cudaMallocManaged(&sphere2, sizeof(Sphere)));
    checkCudaErrors(cudaMallocManaged(&sphere2->mat_ptr, sizeof(Material)));
    m_World->Add(new(sphere2) Sphere(Vec3(1.0f, 0.0f, -1.0f), 0.5f, new(sphere2->mat_ptr) Material(Vec3(0.8f, 0.6f, 0.2f), 0.0f, Mat::metal)));

    Sphere* glassSphere_a;
    checkCudaErrors(cudaMallocManaged(&glassSphere_a, sizeof(Sphere)));
    checkCudaErrors(cudaMallocManaged(&glassSphere_a->mat_ptr, sizeof(Material)));
    m_World->Add(new(glassSphere_a) Sphere(Vec3(-1.0f, 0.0f, -1.0f), 0.5f, new(glassSphere_a->mat_ptr) Material(1.5f, Mat::dielectric)));

    Sphere* glassSphere_b;
    checkCudaErrors(cudaMallocManaged(&glassSphere_b, sizeof(Sphere)));
    checkCudaErrors(cudaMallocManaged(&glassSphere_b->mat_ptr, sizeof(Material)));
    m_World->Add(new(glassSphere_b) Sphere(Vec3(-1.0f, 0.0f, -1.0f), -0.45f, new(glassSphere_b->mat_ptr) Material(1.5f, Mat::dielectric)));

    // checkCudaErrors(cudaMallocManaged(&m_Tree, sizeof(BVHNode)));
    // m_Tree->type = BVH_NODE;
    // m_Tree = new(m_Tree) BVHNode(m_World->objects);
}

void CudaLayer::RunCudaUpdate()
{
    LaunchKernel(static_cast<unsigned int *>(m_CudaDevRenderBuffer), m_ImageWidth, m_ImageHeight, m_SamplesPerPixel, m_MaxDepth, m_World, m_DrandState, m_Inputs);

    // We want to copy cuda_dev_render_buffer data to the texture.
    // Map buffer objects to get CUDA device pointers.
    cudaArray *texture_ptr;
    checkCudaErrors(cudaGraphicsMapResources(1, &m_CudaTexResource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, m_CudaTexResource, 0, 0));

    checkCudaErrors(cudaMemcpy2DToArray(texture_ptr, 0, 0, m_CudaDevRenderBuffer, m_ImageWidth * sizeof(unsigned int), m_ImageWidth * sizeof(unsigned int), m_ImageHeight, cudaMemcpyDeviceToDevice));
    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, &m_CudaTexResource, 0));
}