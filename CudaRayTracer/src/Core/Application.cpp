#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include "Core/Application.h"
#include "Core/Core.h"
#include "Core/Log.h"

#include "Renderer/VertexArray.h"

Application* Application::s_Instance = nullptr;

Application::Application()
{
    RT_ASSERT(!s_Instance, "Application already exists!");
    s_Instance = this;

    m_Window = std::unique_ptr<Window>(Window::Create());

    m_ImGuiLayer = new ImGuiLayer();
    PushOverlay(m_ImGuiLayer);

}

Application::~Application()
{
}

void Application::PushLayer(Layer *layer)
{
    m_LayerStack.PushLayer(layer);
    layer->OnAttach();
}

void Application::PushOverlay(Layer *layer)
{
    m_LayerStack.PushOverlay(layer);
    layer->OnAttach();
}

void Application::Run()
{
    while (!glfwWindowShouldClose(m_Window->GetNativeWindow()))
    {
        m_Window->OnUpdate();
    }
}
