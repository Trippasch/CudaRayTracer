#include "Application.h"

#include "Utils/glfw_tools.h"
#include "Utils/gl_tools.h"

#include <GLFW/glfw3.h>

#define IMGUI_IMPL_API
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

Application* Application::s_Instance = nullptr;

Application::Application()
{
    s_Instance = this;

    m_Window = std::unique_ptr<Window>(Window::Create());
    printGLFWInfo(m_Window->GetNativeWindow());
    printGLInfo();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;   // Enable Docking

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
    ImGuiStyle &style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Setup Platform/Rendere backends
    ImGui_ImplGlfw_InitForOpenGL(m_Window->GetNativeWindow(), true);
    ImGui_ImplOpenGL3_Init("#version 460");
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
