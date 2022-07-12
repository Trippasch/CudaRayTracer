#include "Application.h"
#include "Utils/glfw_tools.h"
#include "Utils/gl_tools.h"

#include <GLFW/glfw3.h>

Application* Application::s_Instance = nullptr;

Application::Application()
{
    s_Instance = this;

    m_Window = std::unique_ptr<Window>(Window::Create());
    printGLFWInfo(m_Window->GetNativeWindow());
    printGLInfo();
}

Application::~Application()
{
}

void Application::Run()
{
    while (!glfwWindowShouldClose(m_Window->GetNativeWindow()))
    {
        m_Window->OnUpdate();
    }
}
