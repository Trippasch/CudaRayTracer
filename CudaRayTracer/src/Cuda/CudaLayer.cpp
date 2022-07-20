#include "Cuda/CudaLayer.h"

#include <cuda_runtime.h>
#include "Utils/helper_cuda.h"

CudaLayer::CudaLayer()
    : Layer("CudaLayer")
{
}

void CudaLayer::OnAttach()
{
    findCudaDevice();
}

void CudaLayer::OnDetach()
{

}

void CudaLayer::OnUpdate()
{

}

void CudaLayer::OnImGuiRender()
{

}