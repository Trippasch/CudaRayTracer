#include "Core/Window.h"

#include <glad/glad.h>

#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui.h>

#include "Core/Core.h"
#include "Core/Log.h"

#include "Utils/gl_tools.h"
#include "Utils/glfw_tools.h"

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

    if (!s_GLFWInitialized) {
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
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    glfwSetWindowPos(m_Window, (mode->width - m_Data.Width) / 2, (mode->height - m_Data.Height) / 2);

    /* Make the window's context curren */
    glfwMakeContextCurrent(m_Window);
    int status = gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    RT_ASSERT(status, "Could not initialize Glad.");

    printGLFWInfo(GetNativeWindow());
    printGLInfo();

    SetVSync(false);

    glfwSetWindowUserPointer(m_Window, this);

    // Set GLFW callbacks
    glfwSetKeyCallback(m_Window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        Window& w = *(Window*)glfwGetWindowUserPointer(window);

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }

        if (glfwGetKey(window, GLFW_KEY_F11) == GLFW_PRESS) {

            GLFWmonitor* monitor = glfwGetWindowMonitor(window);
            if (monitor != nullptr) {
                glfwSetWindowMonitor(window, nullptr, w.GetXPos(), w.GetYPos(), w.GetWidth(), w.GetHeight(), 0);
            }
            else {
                int xpos, ypos, width, height;
                glfwGetWindowPos(window, &xpos, &ypos);
                glfwGetWindowSize(window, &width, &height);
                w.SetWidth(width);
                w.SetHeight(height);
                w.SetXPos(xpos);
                w.SetYPos(ypos);
                GLFWmonitor* monitor = glfwGetPrimaryMonitor();
                const GLFWvidmode* mode = glfwGetVideoMode(monitor);
                glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
            }
        }

        if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) {
            if (!w.m_PauseRender)
                w.m_PauseRender = true;
            else
                w.m_PauseRender = false;
        }
    });

    glfwSetFramebufferSizeCallback(m_Window, [](GLFWwindow* window, int width, int height) {
        RT_TRACE("Resizing window to {0}x{1}", width, height);
    });
}

void Window::Shutdown()
{
    glfwDestroyWindow(m_Window);
    glfwTerminate();
}

void Window::OnUpdate()
{
    /* Swap front and back buffers */
    glfwSwapBuffers(m_Window);
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
