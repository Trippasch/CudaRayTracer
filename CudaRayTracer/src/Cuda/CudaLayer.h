#pragma once

#include "Core/Layer.h"

#include "Hittables/Hittable.cuh"
#include "Hittables/Material.cuh"

#include "Utils/SharedStructs.h"
#include "Utils/helper_cuda.h"

#include "Renderer/Camera.h"

#include <glad/glad.h>

// Cuda
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

class CudaLayer : public Layer
{
public:
    CudaLayer();
    ~CudaLayer() = default;

    virtual void OnAttach() override;
    virtual void OnUpdate() override;
    virtual void OnDetach() override;
    virtual void OnImGuiRender() override;

    inline int GetImageWidth() const
    {
        return m_ImageWidth;
    }
    inline int GetImageHeight() const
    {
        return m_ImageHeight;
    }
    inline void SetImageWidth(int width)
    {
        m_ImageWidth = width;
    }
    inline void SetImageHeight(int height)
    {
        m_ImageHeight = height;
    }

private:
    void InitCudaBuffers();
    void InitGLBuffers();
    void RunCudaInit();
    void GenerateWorld();
    void RunCudaUpdate();
    bool OnImGuiResize();
    void MaterialNode(Material* material, size_t i);
    void TextureNode(Texture* texture, size_t i);
    void ImageAllocation(Image* image);
    void AddHittable();
    void DeleteHittable(Hittable* hittable, int i);
    void DeleteImageAllocation(Hittable* hittable);
    void ClearScene();

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
    curandState* m_DrandState; // allocate random state
    curandState* m_DrandState2;

    InputStruct m_Inputs;

    // Hittables
    Hittable* m_World;
    Hittable** m_List;
    size_t m_ListSize;
    char* m_ListMemory;
    char* m_WorldMemory;
    char* temp = nullptr;

    size_t m_TotalWorldSize;
    size_t m_TotalSize;

    size_t m_ConstantSize;
    size_t m_CheckerSize;

    size_t m_LambertianSize;
    size_t m_MetalSize;
    size_t m_DielectricSize;
    size_t m_DiffuseSize;

    size_t m_SphereSize;
    size_t m_XYrectSize;
    size_t m_XZrectSize;
    size_t m_YZrectSize;

    size_t m_SpheresSize;
    size_t m_GroundSize;

    std::list<int> m_InactiveHittables;

    bool m_UseHittableSphere = true;
    bool m_UseHittableXYRect = false;
    bool m_UseHittableXZRect = false;
    bool m_UseHittableYZRect = false;
    int m_HittableID = 1;
    Vec3 m_HittablePosition = Vec3(0.0f, 1.0f, 0.0f);
    float m_SphereRadius = 0.5f;
    float m_RectWidth = 2.0f;
    float m_RectHeight = 2.0f;

    // RayTracing
    unsigned int m_SamplesPerPixel = 1;
    unsigned int m_MaxDepth = 12;

    // Light properties
    int m_LightIntensity = 2;

    // Texture Image
    int m_TextureImageWidth;
    int m_TextureImageHeight;
    int m_TextureImageNR;
    unsigned char* m_TextureImageData = nullptr;
    char* m_TextureImageFilename = nullptr;

    // hittable id when openning the file dialog
    int m_ButtonID = -1;

    // Camera
    std::unique_ptr<Camera> m_Camera;

    // Background
    Vec3 m_BackgroundStart = Vec3(1.0f, 1.0f, 1.0f);
    Vec3 m_BackgroundEnd = Vec3(0.5f, 0.7f, 1.0f);

    bool m_Paused = false;
};
