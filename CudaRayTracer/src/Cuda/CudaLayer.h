#pragma once

#include "Core/Layer.h"

class CudaLayer : public Layer
{
public:
    CudaLayer();
    ~CudaLayer() = default;

    virtual void OnAttach() override;
    virtual void OnUpdate() override;
    virtual void OnDetach() override;
    virtual void OnImGuiRender() override;
};