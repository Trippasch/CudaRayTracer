#include "Application.h"

#include <GLFW/glfw3.h>

Application* Application::s_Instance = nullptr;

Application::Application()
{
    s_Instance = this;

    m_Window = std::unique_ptr<Window>(Window::Create());
}

Application::~Application()
{
}

void Application::Run()
{
    while (!glfwWindowShouldClose((GLFWwindow*)m_Window->GetNativeWindow()))
    {
        m_Window->OnUpdate();
    }
}
