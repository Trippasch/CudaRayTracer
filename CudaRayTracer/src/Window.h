#pragma once

#include <string>

#include <GLFW/glfw3.h>

struct WindowProps
{
    std::string Title;
    uint32_t Width;
    uint32_t Height;

    WindowProps(const std::string& title = "CudaRayTracer",
        uint32_t width = 1280,
        uint32_t height = 720)
        : Title(title), Width(width), Height(height) {}
};

class Window
{
public:
    Window(const WindowProps& props);
    ~Window();

    void OnUpdate();

    inline uint32_t GetWidth() const { return m_Data.Width; }
    inline uint32_t GetHeight() const { return m_Data.Height; }

    // Window attributes
    // inline void SetEventCallback(const EventCallbackFn& callback) override { m_Data.EventCallback = callback; }
    void SetVSync(bool enabled);
    bool IsVSync() const;

    inline virtual void* GetNativeWindow() const { return m_Window; }

    static Window* Create(const WindowProps& props = WindowProps());
private:
    virtual void Init(const WindowProps& props);
    virtual void Shutdown();
private:
    GLFWwindow* m_Window;

    struct WindowData
    {
        std::string Title;
        unsigned int Width, Height;
        bool VSync;

        // EventCallbackFn EventCallback;
    };

    WindowData m_Data;
};
