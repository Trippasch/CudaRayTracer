#include "Window.h"

#include <glad/glad.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include "Log.h"
#include "Core.h"

static bool s_GLFWInitialized = false;

 static void GLFWErrorCallback(int error, const char* description)
 {
     RT_ERROR("GLFW Error {0}: {1}", error, description);
 }

Window* Window::Create(const WindowProps& props)
{
    return new Window(props);
}

Window::Window(const WindowProps& props)
{
    Init(props);
}

Window::~Window()
{
    Shutdown();
}

void Window::Init(const WindowProps& props)
{
    m_Data.Title = props.Title;
    m_Data.Width = props.Width;
    m_Data.Height = props.Height;

     RT_INFO("Creating window {0} ({1}, {2})", props.Title, props.Width, props.Height);

    if (!s_GLFWInitialized)
    {
        // TODO: glfwTerminate on system shutdown
        int success = glfwInit();
        RT_ASSERT(success, "Could not initialize GLFW!");
        glfwSetErrorCallback(GLFWErrorCallback);
        s_GLFWInitialized = true;
    }

    // These hints switch the OpenGL profile to core
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    m_Window = glfwCreateWindow((int)props.Width, (int)props.Height, m_Data.Title.c_str(), nullptr, nullptr);
    RT_ASSERT(m_Window, "Could not create GLFW window");

    /* Set the window's position */
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode *mode = glfwGetVideoMode(monitor);
    glfwSetWindowPos(m_Window, (mode->width - m_Data.Width) / 2, (mode->height - m_Data.Height) / 2);

    /* Make the window's context curren */
    glfwMakeContextCurrent(m_Window);
    int status = gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    RT_ASSERT(status, "Could not initialize Glad.");

    glfwSetWindowUserPointer(m_Window, &m_Data);
    SetVSync(true);

    // Set GLFW callbacks
    glfwSetKeyCallback(m_Window, [](GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, true);
        }
    });

    glfwSetFramebufferSizeCallback(m_Window, [](GLFWwindow* window, int width, int height)
    {
        // make sure the viewport matches the new window dimensions; note that width and
        // height will be significantly larger than specified on retina displays.
        RT_TRACE("Resizing window to {0}x{1}", width, height);
        glViewport(0, 0, width, height);
    });

    // glfwSetWindowSizeCallback(m_Window, [](GLFWwindow* window, int width, int height)
    //     {
    //         WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
    //         data.Width = width;
    //         data.Height = height;

    //         WindowResizeEvent event(width, height);
    //         data.EventCallback(event);
    //     });

    // glfwSetWindowCloseCallback(m_Window, [](GLFWwindow* window)
    //     {
    //         WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

    //         WindowCloseEvent event;
    //         data.EventCallback(event);
    //     });

    // glfwSetKeyCallback(m_Window, [](GLFWwindow* window, int key, int scancode, int action, int mods)
    //     {
    //         WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

    //         switch (action)
    //         {
    //             case GLFW_PRESS:
    //             {
    //                 KeyPressedEvent event(key, 0);
    //                 data.EventCallback(event);
    //                 break;
    //             }
    //             case GLFW_RELEASE:
    //             {
    //                 KeyReleasedEvent event(key);
    //                 data.EventCallback(event);
    //                 break;
    //             }
    //             case GLFW_REPEAT:
    //             {
    //                 KeyPressedEvent event(key, 1);
    //                 data.EventCallback(event);
    //                 break;
    //             }
    //         }
    //     });

    // glfwSetCharCallback(m_Window, [](GLFWwindow* window, unsigned int keycode)
    //     {
    //         WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

    //         KeyTypedEvent event(keycode);
    //         data.EventCallback(event);
    //     });

    // glfwSetMouseButtonCallback(m_Window, [](GLFWwindow* window, int button, int action, int mods)
    //     {
    //         WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

    //         switch (action)
    //         {
    //             case GLFW_PRESS:
    //             {
    //                 MouseButtonPressedEvent event(button);
    //                 data.EventCallback(event);
    //                 break;
    //             }
    //             case GLFW_RELEASE:
    //             {
    //                 MouseButtonReleasedEvent event(button);
    //                 data.EventCallback(event);
    //                 break;
    //             }
    //         }
    //     });

    // glfwSetScrollCallback(m_Window, [](GLFWwindow* window, double xOffset, double yOffset)
    //     {
    //         WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

    //         MouseScrolledEvent event((float)xOffset, (float)yOffset);
    //         data.EventCallback(event);
    //     });

    // glfwSetCursorPosCallback(m_Window, [](GLFWwindow* window, double xPos, double yPos)
    //     {
    //         WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);

    //         MouseMovedEvent event((float)xPos, (float)yPos);
    //         data.EventCallback(event);
    //     });
}

void Window::Shutdown()
{
    glfwDestroyWindow(m_Window);
}

void Window::OnUpdate()
{
    /* Render here */
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    /*****************ImGui *****************/
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    {
        static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

        // We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
        // because it would be confusing to have two docking targets within each others.
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;
        // if (m_MenubarCallback)
        //     window_flags |= ImGuiWindowFlags_MenuBar;

        const ImGuiViewport *viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->WorkPos);
        ImGui::SetNextWindowSize(viewport->WorkSize);
        ImGui::SetNextWindowViewport(viewport->ID);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
        window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

        // When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background
        // and handle the pass-thru hole, so we ask Begin() to not render a background.
        if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
            window_flags |= ImGuiWindowFlags_NoBackground;

        // Important: note that we proceed even if Begin() returns false (aka window is collapsed).
        // This is because we want to keep our DockSpace() active. If a DockSpace() is inactive,
        // all active windows docked into it will lose their parent and become undocked.
        // We cannot preserve the docking relationship between an active window and an inactive docking, otherwise
        // any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::Begin("DockSpace Demo", nullptr, window_flags);
        ImGui::PopStyleVar();

        ImGui::PopStyleVar(2);

        // Submit the DockSpace
        ImGuiIO &io = ImGui::GetIO();
        if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
        {
            ImGuiID dockspace_id = ImGui::GetID("VulkanAppDockspace");
            ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
        }

        ImGui::ShowDemoWindow();

        ImGui::Begin("Hello");
        ImGui::End();

        // if (m_MenubarCallback)
        // {
        //     if (ImGui::BeginMenuBar())
        //     {
        //         m_MenubarCallback();
        //         ImGui::EndMenuBar();
        //     }
        // }

        // for (auto &layer : m_LayerStack)
        //     layer->OnUIRender();

        ImGui::End();
    }

    // Rendering
    ImGui::Render();
    ImGuiIO &io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        GLFWwindow *backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    /****************************************/

    /* Swap front and back buffers */
    glfwSwapBuffers(m_Window);
    // m_Context->SwapBuffers();

    /* Poll for and process events */
    glfwPollEvents();
}

void Window::SetVSync(bool enabled)
{
    if (enabled)
        glfwSwapInterval(1);
    else
        glfwSwapInterval(0);

    m_Data.VSync = enabled;
}

bool Window::IsVSync() const
{
    return m_Data.VSync;
}
