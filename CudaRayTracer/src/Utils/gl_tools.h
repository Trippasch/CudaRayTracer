#pragma once

#include <glad/glad.h>

#include "Core/Log.h"

void printGLInfo() {

    RT_INFO("OpenGL: GL version: {0}", (const char*)glGetString(GL_VERSION));
    RT_INFO("OpenGL: GLSL version: {0}", (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));
    RT_INFO("OpenGL: Vendor: {0}", (const char*)glGetString(GL_VENDOR));
    RT_INFO("OpenGL: Renderer: {0}", (const char*)glGetString(GL_RENDERER));
}