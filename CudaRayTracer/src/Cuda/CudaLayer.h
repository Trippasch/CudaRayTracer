#pragma once

#include "Core/Layer.h"

#include "Hittables/HittableList.cuh"
#include "Hittables/Material.cuh"
#include "Hittables/Hittable.cuh"

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
    void AddHittable();
    void DeleteHittable(Hittable* hittable);
    void RunCudaUpdate();
    bool OnImGuiResize();

private:
    // Image
    unsigned int m_ImageWidth = 800;
    unsigned int m_ImageHeight = 600;

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

    InputStruct m_Inputs;

    // Hittables
    HittableList* m_World;
    bool m_UseHittableSphere = true;
    bool m_UseHittableXYRect = false;
    bool m_UseHittableXZRect = false;
    bool m_UseHittableYZRect = false;
    int m_HittableID = 0;
    Vec3 m_SpherePosition = Vec3(0.0f, 1.0f, 0.0f);
    float m_SphereRadius = 0.5f;
    Vec3 m_RectPosition = Vec3(0.0f, 1.25f, -3.0f);
    float m_RectWidth = 6.0f;
    float m_RectHeight = 3.5f;

    // RayTracing
    unsigned int m_SamplesPerPixel = 50;
    unsigned int m_MaxDepth = 12;

    // Material Properties
    bool m_UseLambertian = true;
    bool m_UseMetal = false;
    bool m_UseDielectric = false;
    bool m_UseDiffuseLight = false;
    bool m_UseConstantTexture = true;
    bool m_UseCheckerTexture = false;
    bool m_UseImageTexture = false;
    Vec3 m_newColor = Vec3(1.0f, 1.0f, 1.0f);
    float m_Fuzz = 0.0f;
    float m_IR = 0.0f;

    // Light properties
    int m_LightIntensity = 2;

    // Texture Image
    int m_TextureImageWidth;
    int m_TextureImageHeight;
    int m_TextureImageNR;
    unsigned char* m_TextureImageData = nullptr;
    const char* m_TextureImageFilename = nullptr;

    // Camera
    std::unique_ptr<Camera> m_Camera;
};