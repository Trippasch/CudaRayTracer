#include "Cuda/CudaLayer.h"

#include <cuda_runtime.h>
#include <curand_kernel.h> 
#include <cuda_gl_interop.h>

#include <imgui.h>

#include "Utils/helper_cuda.h"

// Forward declaration of CUDA render
extern "C"
void launch_cudaRender(dim3 grid, dim3 block, int sbytes, unsigned int *g_odata, int imgw);

CudaLayer::CudaLayer()
    : Layer("CudaLayer")
{
}

void CudaLayer::OnAttach()
{
    findCudaDevice();

    InitCudaBuffers();
    InitGLBuffers();
}

void CudaLayer::OnDetach()
{
    checkCudaErrors(cudaFree(m_CudaDevRenderBuffer));
    checkCudaErrors(cudaGraphicsUnregisterResource(m_CudaTexResource));
    // useful for compute-sanitizer --leak-check full
    cudaDeviceReset();
}

void CudaLayer::OnUpdate()
{
    dim3 block(16, 16, 1);
    dim3 grid(m_ImageWidth / block.x, m_ImageHeight / block.y, 1);

    launch_cudaRender(grid, block, 0, static_cast<unsigned int *>(m_CudaDevRenderBuffer), m_ImageWidth);

    // We want to copy cuda_dev_render_buffer data to the texture.
    // Map buffer objects to get CUDA device pointers.
    cudaArray *texture_ptr;
    checkCudaErrors(cudaGraphicsMapResources(1, &m_CudaTexResource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, m_CudaTexResource, 0, 0));

    checkCudaErrors(cudaMemcpyToArray(texture_ptr, 0, 0, m_CudaDevRenderBuffer, m_SizeTexData, cudaMemcpyDeviceToDevice));
    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, &m_CudaTexResource, 0));
}

void CudaLayer::OnImGuiRender()
{
    ImGui::Begin("Generated Image");
    ImGui::Image((ImTextureID)m_OpenGLTexCuda, {static_cast<float>(m_ImageWidth), static_cast<float>(m_ImageHeight)}, ImVec2(0, 1), ImVec2(1, 0));
    ImGui::End();
}

void CudaLayer::InitCudaBuffers()
{
    // We don't want to use cudaMallocManaged here - since we definitely want
    checkCudaErrors(cudaMalloc(&m_CudaDevRenderBuffer, m_SizeTexData)); // Allocate CUDA memory for color output
    // // Allocate random state
    // checkCudaErrors(cudaMalloc((void **)&m_DrandState, m_NumTexels * sizeof(curandState)));
    // checkCudaErrors(cudaMalloc((void **)&m_DrandState2, 1 * sizeof(curandState)));
    // // Allocate hittables
    // checkCudaErrors(cudaMalloc((void **)&d_list, num_hittables * sizeof(hittable *)));
    // checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
}

void CudaLayer::InitGLBuffers()
{
    // create an OpenGL texture
    glGenTextures(1, &m_OpenGLTexCuda);              // generate 1 texture
    glBindTexture(GL_TEXTURE_2D, m_OpenGLTexCuda); // set it as current target
    // set basic texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // Specify 2D texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI, m_ImageWidth, m_ImageHeight, 0, GL_RGB_INTEGER, GL_UNSIGNED_BYTE, NULL);
    // Register this texture with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterImage(&m_CudaTexResource, m_OpenGLTexCuda, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    // SDK_CHECK_ERROR_GL();
}