#pragma once

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class Camera
{
public:
    Camera(int width, int height, glm::vec3 position);

    inline float GetWidth() { return this->m_Width; }
    inline void SetWidth(float width) { this->m_Width = width; }

    inline float GetHeight() { return this->m_Height; }
    inline void SetHeight(float height) { this->m_Height = height; }

    glm::mat4 Matrix(float fovDeg, float nearPlane, float farPlane);

    // Handles camera inputs
    void Inputs(GLFWwindow* window);

public:
    glm::vec3 m_Position;
    glm::vec3 m_Orientation = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 m_Up = glm::vec3(0.0f, 1.0f, 0.0f);

private:
    // Prevents the camera from jumping around when first clicking left click
    bool m_FirstClick = true;

    // Stores the width and height of the window
    float m_Width;
    float m_Height;

    // Adjust the speed of the camera and it's sensitivity when looking around
    float m_Speed = 0.01f;
    float m_Sensitivity = 100.0f;
};