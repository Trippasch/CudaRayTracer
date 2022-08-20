#include "Cuda/CudaLayer.h"

#include <imgui.h>

// Forward declaration of CUDA render
// extern "C"
// void launch_cudaRender(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int imgw);

extern "C"
void launch_kernel(dim3 grid, dim3 block, int sbytes, unsigned int *pos, unsigned int window_width, unsigned int window_height, const unsigned int samples_per_pixel, const unsigned int max_depth, hittable **d_world, curandState *d_rand_state, InputStruct inputs);

extern "C" void launch_rand_init(curandState *d_rand_state2);

extern "C"
void launch_render_init(dim3 grid, dim3 block, unsigned int window_width, unsigned int window_height, curandState *d_rand_state);

extern "C"
void launch_create_world(hittable **d_list, hittable **d_world, const float aspect_ratio, curandState *d_rand_state2);

extern "C"
void launch_free_world(hittable **d_list, hittable **d_world, const unsigned int num_hittables);

CudaLayer::CudaLayer()
    : Layer("CudaLayer")
{
}

void CudaLayer::OnAttach()
{
    findCudaDevice();

    InitCudaBuffers();
    InitGLBuffers();

    RunCudaInit();
    m_Camera = std::make_unique<Camera>(m_ImageWidth, m_ImageHeight, glm::vec3(0.0f, 0.0f, 2.0f));
}

void CudaLayer::OnDetach()
{
    launch_free_world(m_HittableList, m_World, m_NumHittables);
    checkCudaErrors(cudaFree(m_CudaDevRenderBuffer));
    checkCudaErrors(cudaFree(m_HittableList));
    checkCudaErrors(cudaFree(m_World));
    checkCudaErrors(cudaFree(m_DrandState));
    checkCudaErrors(cudaFree(m_DrandState2));
    checkCudaErrors(cudaGraphicsUnregisterResource(m_CudaTexResource));

    // useful for compute-sanitizer --leak-check full
    cudaDeviceReset();
}

void CudaLayer::OnUpdate()
{
    // m_Camera->Inputs(m_Window);

    m_Inputs.origin_x = m_Camera->m_Position.x;
    m_Inputs.origin_y = m_Camera->m_Position.y;
    m_Inputs.origin_z = m_Camera->m_Position.z;

    m_Inputs.orientation_x = m_Camera->m_Orientation.x;
    m_Inputs.orientation_y = m_Camera->m_Orientation.y;
    m_Inputs.orientation_z = m_Camera->m_Orientation.z;

    m_Inputs.up_x = m_Camera->m_Up.x;
    m_Inputs.up_y = m_Camera->m_Up.y;
    m_Inputs.up_z = m_Camera->m_Up.z;

    RunCudaUpdate();
}

void CudaLayer::OnImGuiRender()
{
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Generated Image");

    float viewport_width = ImGui::GetContentRegionAvail().x;
    float viewport_height = ImGui::GetContentRegionAvail().y;

    ImGui::Image((void*)(intptr_t)m_Texture, ImVec2(viewport_width, viewport_height));
    ImGui::PopStyleVar();
    ImGui::End();
}

void CudaLayer::InitCudaBuffers()
{
    // We don't want to use cudaMallocManaged here - since we definitely want
    checkCudaErrors(cudaMalloc(&m_CudaDevRenderBuffer, m_SizeTexData)); // Allocate CUDA memory for color output
    // Allocate random state
    checkCudaErrors(cudaMalloc((void **)&m_DrandState, m_NumTexels * sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void **)&m_DrandState2, 1 * sizeof(curandState)));
    // Allocate hittables
    checkCudaErrors(cudaMalloc((void **)&m_HittableList, m_NumHittables * sizeof(hittable *)));
    checkCudaErrors(cudaMalloc((void **)&m_World, sizeof(hittable *)));
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
    launch_rand_init(m_DrandState2);

    dim3 block(16, 16, 1);
    dim3 grid(m_ImageWidth / block.x, m_ImageHeight / block.y, 1);

    launch_render_init(grid, block, m_ImageWidth, m_ImageHeight, m_DrandState);

    launch_create_world(m_HittableList, m_World, m_AspectRatio, m_DrandState2);
}

void CudaLayer::RunCudaUpdate()
{
    dim3 block(16, 16, 1);
    dim3 grid(m_ImageWidth / block.x, m_ImageHeight / block.y, 1);

    // launch_cudaRender(grid, block, 0, static_cast<unsigned int *>(m_CudaDevRenderBuffer), m_ImageWidth);

    launch_kernel(grid, block, 0, static_cast<unsigned int *>(m_CudaDevRenderBuffer), m_ImageWidth, m_ImageHeight, m_SamplesPerPixel, m_MaxDepth, m_World, m_DrandState, m_Inputs);

    // We want to copy cuda_dev_render_buffer data to the texture.
    // Map buffer objects to get CUDA device pointers.
    cudaArray *texture_ptr;
    checkCudaErrors(cudaGraphicsMapResources(1, &m_CudaTexResource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, m_CudaTexResource, 0, 0));

    checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, m_CudaDevRenderBuffer, m_SizeTexData, cudaMemcpyDeviceToDevice));
    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, &m_CudaTexResource, 0));
}