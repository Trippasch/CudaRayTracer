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

    //m_Inputs.far = m_Camera->m_FarPlane;
    //m_Inputs.near = m_Camera->m_NearPlane;
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

    float viewport_width = ImGui::GetContentRegionAvail().x;
    float viewport_height = ImGui::GetContentRegionAvail().y;

    ImGui::Image((void*)(intptr_t)m_Texture, ImVec2(viewport_width, viewport_height), ImVec2(0, 1), ImVec2(1, 0));
    ImGui::PopStyleVar();

    // IsWindowFocused() has a minor bug -- it centers the mouse when losing focus
    if (ImGui::IsWindowFocused()) {

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

void CudaLayer::InitCudaBuffers()
{
    // We don't want to use cudaMallocManaged here - since we definitely want
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
    // Sphere* groundSphere;
    // checkCudaErrors(cudaMallocManaged(&groundSphere, sizeof(Sphere)));
    // checkCudaErrors(cudaMallocManaged(&groundSphere->mat_ptr, sizeof(Lambertian)));
    // m_World->Add(new(groundSphere) Sphere(Vec3(0.0f, -100.5f, 0.0f), 100.0f, new(groundSphere->mat_ptr) Lambertian(Vec3(0.8f, 0.8f, 0.0f))));

    // Sphere* sphere1;
    // checkCudaErrors(cudaMallocManaged(&sphere1, sizeof(Sphere)));
    // checkCudaErrors(cudaMallocManaged(&sphere1->mat_ptr, sizeof(Lambertian)));
    // m_World->Add(new(sphere1) Sphere(Vec3(0.0f, 0.0f, -1.0f), 0.5f, new(sphere1->mat_ptr) Lambertian(Vec3(0.1f, 0.2f, 0.5f))));

    auto material1 = new Lambertian(Vec3(0.8f, 0.8f, 0.0f));
    Sphere* groundSphere = new Sphere(Vec3(0.0f, -100.5f, 0.0f), 100.0f, material1);
    m_World->Add(groundSphere);

    auto material2 = new Lambertian(Vec3(0.1f, 0.2f, 0.5f));
    Sphere* sphere = new Sphere(Vec3(0.0f, 0.0f, -1.0f), 0.5f, material2);
    m_World->Add(sphere);
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