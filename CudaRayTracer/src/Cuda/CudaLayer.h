#pragma once

#include "Core/Layer.h"

#include <glad/glad.h>

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

private:
    void InitCudaBuffers();
    void InitGLBuffers();

private:
    // Image
    const float m_AspectRatio = 4.0f / 3.0f;
    const int m_ImageWidth = 800;
    const int m_ImageHeight = static_cast<int>(m_ImageWidth / m_AspectRatio);

    // Cuda Image
    unsigned int m_NumTexels = m_ImageWidth * m_ImageHeight;
    unsigned int m_NumValues = m_NumTexels * 4;
    size_t m_SizeTexData = sizeof(GLuint) * m_NumValues;

    // Cuda-OpenGL interops
    struct cudaGraphicsResource *m_CudaTexResource;
    GLuint m_OpenGLTexCuda;
    void *m_CudaDevRenderBuffer;
    // curandState *m_DrandState;    // allocate random state
    // curandState *m_DrandState2;
};