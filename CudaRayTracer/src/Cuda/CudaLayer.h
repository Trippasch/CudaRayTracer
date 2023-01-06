#pragma once

#include "Core/Layer.h"
// #include "Hittables/BVHNode.h"
#include "Hittables/HittableList.h"
#include "Hittables/Material.h"
#include "Hittables/Sphere.h"
#include "Utils/SharedStructs.h"
#include "Utils/helper_cuda.h"

#include "Renderer/Camera.h"

#include <glad/glad.h>

// Cuda
#include <cuda_runtime.h>
#include <curand_kernel.h> 
#include <cuda_gl_interop.h>

class CudaLayer : public Layer
{
public:
    CudaLayer();
    ~CudaLayer() = default;

    virtual void OnAttach() override;
    virtual void OnUpdate() override;
    virtual void OnDetach() override;
    virtual void OnImGuiRender() override;

    inline int GetImageWidth() const { return m_ImageWidth; }
    inline int GetImageHeight() const { return m_ImageHeight; }
    inline void SetImageWidth(int width) { m_ImageWidth = width; }
    inline void SetImageHeight(int height) { m_ImageHeight = height; }

private:
    void InitCudaBuffers();
    void InitGLBuffers();
    void RunCudaInit();
    void GenerateWorld();
    void RunCudaUpdate();
    bool onImGuiResize();

private:
    // Image
    const float m_AspectRatio = 4.0f / 3.0f;
    int m_ImageWidth = 800;
    int m_ImageHeight = static_cast<int>(m_ImageWidth / m_AspectRatio);

    // Cuda Image
    unsigned int m_NumTexels = m_ImageWidth * m_ImageHeight;
    unsigned int m_NumValues = m_NumTexels * 4;
    size_t m_SizeTexData = sizeof(GLubyte) * m_NumValues;

    // Cuda-OpenGL interops
    struct cudaGraphicsResource* m_CudaTexResource;
    void* m_CudaDevRenderBuffer;
    GLuint m_Texture;
    curandState* m_DrandState;    // allocate random state
    curandState* m_DrandState2;

    // Hittables
    HittableList* m_World;

    // RayTracing
    unsigned int m_SamplesPerPixel = 10;
    unsigned int m_MaxDepth = 10;

    InputStruct m_Inputs;

    // Camera
    std::unique_ptr<Camera> m_Camera;
};