#include "Cuda/CudaLayer.h"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include "../Utils/RawStbImage.h"

extern "C"
void LaunchKernel(unsigned int *pos, unsigned int image_width, unsigned int image_height, const unsigned int samples_per_pixel, const unsigned int max_depth, HittableList* world, curandState *d_rand_state, InputStruct inputs);

extern "C" void LaunchRandInit(curandState *d_rand_state2);

extern "C"
void LaunchRenderInit(dim3 grid, dim3 block, unsigned int image_width, unsigned int image_height, curandState *d_rand_state);

// extern "C"
// void LaunchCreateWorld(Sphere **d_list, Sphere **d_world, curandState *d_rand_state2);

// extern "C"
// void LaunchFreeWorld(Sphere **d_list, Sphere **d_world, const unsigned int num_hittables);

CudaLayer::CudaLayer()
    : Layer("CudaLayer")
{
    m_World = new HittableList();
}

void CudaLayer::OnAttach()
{
    findCudaDevice();

    size_t pValue;
    cudaDeviceSetLimit(cudaLimitStackSize, 2048);
    cudaDeviceGetLimit(&pValue, cudaLimitStackSize);
    RT_INFO("CUDA Stack Size Limit: {0} bytes", pValue);

    InitCudaBuffers();
    InitGLBuffers();

    RunCudaInit();

    GenerateWorld();

    m_Camera = std::make_unique<Camera>(m_ImageWidth, m_ImageHeight, glm::vec3(0.0f, 2.0f, 10.0f));

    glm::vec3 rightV = glm::normalize(glm::cross(m_Camera->m_Orientation, m_Camera->m_Up));
    glm::vec3 upV = glm::normalize(glm::cross(m_Camera->m_Orientation, rightV));

    m_Inputs.origin_x = m_Camera->m_Position.x;
    m_Inputs.origin_y = m_Camera->m_Position.y;
    m_Inputs.origin_z = m_Camera->m_Position.z;

    m_Inputs.orientation_x = m_Camera->m_Orientation.x;
    m_Inputs.orientation_y = m_Camera->m_Orientation.y;
    m_Inputs.orientation_z = m_Camera->m_Orientation.z;

    m_Inputs.up_x = upV.x;
    m_Inputs.up_y = upV.y;
    m_Inputs.up_z = upV.z;

    m_Inputs.far_plane = m_Camera->m_FarPlane;
    m_Inputs.near_plane = m_Camera->m_NearPlane;
    m_Inputs.fov = glm::radians(m_Camera->m_Fov);
}

void CudaLayer::OnDetach()
{
    for (auto obj : m_World->objects) {
        DeleteSphere(obj);
    }

    // LaunchFreeWorld(m_HittableList, m_World, m_NumHittables);
    checkCudaErrors(cudaFree(m_CudaDevRenderBuffer));
    checkCudaErrors(cudaFree(m_DrandState));
    checkCudaErrors(cudaFree(m_DrandState2));
    // checkCudaErrors(cudaFree(m_HittableList));
    // checkCudaErrors(cudaFree(m_World));

    checkCudaErrors(cudaGraphicsUnregisterResource(m_CudaTexResource));

    // useful for compute-sanitizer --leak-check full
    cudaDeviceReset();
}

void CudaLayer::OnUpdate()
{
    RunCudaUpdate();
}

void CudaLayer::OnImGuiRender()
{
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("Generated Image");

    if (OnImGuiResize() == false) {
        ImGui::End();
        ImGui::PopStyleVar();
        return;
    }

    ImGui::ImageButton((void*)(intptr_t)m_Texture, ImVec2(m_ImageWidth, m_ImageHeight), ImVec2(0, 1), ImVec2(1, 0), 0);
    ImGui::PopStyleVar();

    if (ImGui::IsWindowHovered()) {
        ImGuiIO& io = ImGui::GetIO();
        if (io.MouseWheel) {
            m_Camera->ProcessMouseScroll(io.MouseWheel);
            m_Inputs.fov = glm::radians(m_Camera->m_Fov);
        }
    }

    if (ImGui::IsWindowFocused()) {

        m_Camera->Inputs((GLFWwindow *)ImGui::GetMainViewport()->PlatformHandle);

        glm::vec3 rightV = glm::normalize(glm::cross(m_Camera->m_Orientation, m_Camera->m_Up));
        glm::vec3 upV = glm::normalize(glm::cross(m_Camera->m_Orientation, rightV));

        m_Inputs.origin_x = m_Camera->m_Position.x;
        m_Inputs.origin_y = m_Camera->m_Position.y;
        m_Inputs.origin_z = m_Camera->m_Position.z;

        m_Inputs.orientation_x = m_Camera->m_Orientation.x;
        m_Inputs.orientation_y = m_Camera->m_Orientation.y;
        m_Inputs.orientation_z = m_Camera->m_Orientation.z;

        m_Inputs.up_x = upV.x;
        m_Inputs.up_y = upV.y;
        m_Inputs.up_z = upV.z;
    }
    else {
        glfwSetInputMode((GLFWwindow *)ImGui::GetMainViewport()->PlatformHandle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        m_Camera->m_FirstClick = true;
    }

    ImGui::End();

    ImGui::Begin("Metrics");
    ImGuiIO &io = ImGui::GetIO();

    ImGui::Text("Dear ImGui %s", ImGui::GetVersion());

#ifdef RT_DEBUG
    ImGui::Text("Running on Debug mode");
#elif RT_RELEASE
    ImGui::Text("Running on Release mode");
#elif RT_DIST
    ImGui::Text("Running on Dist mode");
#endif

    ImGui::Text("Application average\n %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);

    ImGui::End();

    ImGui::Begin("Opions");

    static ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_DefaultOpen;

    if (ImGui::CollapsingHeader("Camera", base_flags))
    {
        if (ImGui::DragFloat3("Position", (float *)&m_Camera->m_Position, 0.01f, -FLT_MAX, FLT_MAX, "%.2f")) {
            m_Inputs.origin_x = m_Camera->m_Position.x;
            m_Inputs.origin_y = m_Camera->m_Position.y;
            m_Inputs.origin_z = m_Camera->m_Position.z;
        }
        if (ImGui::DragFloat3("Orientation", (float *)&m_Camera->m_Orientation, 0.01f, -FLT_MAX, FLT_MAX, "%.2f")) {
            m_Inputs.orientation_x = m_Camera->m_Orientation.x;
            m_Inputs.orientation_y = m_Camera->m_Orientation.y;
            m_Inputs.orientation_z = m_Camera->m_Orientation.z;
        }
        if (ImGui::SliderFloat("Field of view", &m_Camera->m_Fov, 1.0f, 120.0f, "%.f")) {
            m_Inputs.fov = glm::radians(m_Camera->m_Fov);
        }
    }

    if (ImGui::CollapsingHeader("Ray Tracing Settings", base_flags)) {
        ImGui::SliderInt("Samples Per Pixel", (int *)&m_SamplesPerPixel, 1, 100);
        ImGui::SliderInt("Max Depth", (int *)&m_MaxDepth, 1, 50);
    }

    if (ImGui::CollapsingHeader("Spheres Settings", base_flags)) {

        for (int i = 0; i < m_World->objects.size(); i++) {
            if (ImGui::TreeNodeEx(("Sphere " + std::to_string(i)).c_str())) {
                ImGui::DragFloat3(("Sphere Position " + std::to_string(i)).c_str(), (float *)&m_World->objects.at(i)->center, 0.01f, -FLT_MAX, FLT_MAX, "%.2f");
                ImGui::DragFloat(("Sphere Radius " + std::to_string(i)).c_str(), (float *)&m_World->objects.at(i)->radius, 0.01f, -FLT_MAX, FLT_MAX, "%.2f");

                if (ImGui::TreeNodeEx(GetTextForEnum(m_World->objects.at(i)->mat_ptr->material), base_flags)) {

                    if (m_World->objects.at(i)->mat_ptr->material == Mat::lambertian && m_World->objects.at(i)->mat_ptr->albedo->texture == Tex::constant_texture) {
                        ImGui::ColorEdit3(("Albedo " + std::to_string(i)).c_str(), (float *)&m_World->objects.at(i)->mat_ptr->albedo->color);
                    }
                    else if (m_World->objects.at(i)->mat_ptr->material == Mat::lambertian && m_World->objects.at(i)->mat_ptr->albedo->texture == Tex::checker_texture) {
                        ImGui::ColorEdit3(("Albedo odd " + std::to_string(i)).c_str(), (float *)&m_World->objects.at(i)->mat_ptr->albedo->odd->color);
                        ImGui::ColorEdit3(("Albedo even " + std::to_string(i)).c_str(), (float *)&m_World->objects.at(i)->mat_ptr->albedo->even->color);
                    }
                    else if (m_World->objects.at(i)->mat_ptr->material == Mat::metal && m_World->objects.at(i)->mat_ptr->albedo->texture == Tex::constant_texture) {
                        ImGui::ColorEdit3(("Albedo " + std::to_string(i)).c_str(), (float *)&m_World->objects.at(i)->mat_ptr->albedo->color);
                        ImGui::DragFloat(("Fuzziness " + std::to_string(i)).c_str(), (float *)&m_World->objects.at(i)->mat_ptr->fuzz, 0.01f, 0.0f, 1.0f, "%.2f");
                    }
                    else if (m_World->objects.at(i)->mat_ptr->material == Mat::metal && m_World->objects.at(i)->mat_ptr->albedo->texture == Tex::checker_texture) {
                        ImGui::ColorEdit3(("Albedo odd " + std::to_string(i)).c_str(), (float *)&m_World->objects.at(i)->mat_ptr->albedo->odd->color);
                        ImGui::ColorEdit3(("Albedo even " + std::to_string(i)).c_str(), (float *)&m_World->objects.at(i)->mat_ptr->albedo->even->color);
                        ImGui::DragFloat(("Fuzziness " + std::to_string(i)).c_str(), (float *)&m_World->objects.at(i)->mat_ptr->fuzz, 0.01f, 0.0f, 1.0f, "%.2f");
                    }
                    else if (m_World->objects.at(i)->mat_ptr->material == Mat::dielectric) {
                        ImGui::DragFloat(("Index of Refraction " + std::to_string(i)).c_str(), (float *)&m_World->objects.at(i)->mat_ptr->ir, 0.01f, 0.0f, FLT_MAX, "%.2f");
                    }

                    ImGui::TreePop();
                }

                ImGui::TreePop();
            }
        }

        if (ImGui::Button("Add Sphere...")) {
            ImGui::OpenPopup("New Sphere");
        }

        // Always center this window when appearing
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

        if (ImGui::BeginPopupModal("New Sphere", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {

            ImGui::Text("Choose the sphere material:");
            ImGui::Separator();

            if (ImGui::Checkbox("Lambertian", &m_UseLambertian)) {
                m_UseMetal = false;
                m_UseDielectric = false;
            }

            if (ImGui::Checkbox("Metal", &m_UseMetal)) {
                m_UseLambertian = false;
                m_UseDielectric = false;
            }

            if (ImGui::Checkbox("Dielectric", &m_UseDielectric)) {
                m_UseMetal = false;
                m_UseLambertian = false;
            }

            ImGui::Separator();

            if (m_UseLambertian == true || m_UseMetal == true) {
                ImGui::Text("Choose the sphere material texture:");
                ImGui::Separator();
                if (ImGui::Checkbox("Constant Texture", &m_UseConstantTexture))
                    m_UseCheckerTexture = false;
                if (ImGui::Checkbox("Checker Texture", &m_UseCheckerTexture))
                    m_UseConstantTexture = false;
                ImGui::Separator();
            }

            if (ImGui::Button("Add")) {
                AddSphere();
                ImGui::CloseCurrentPopup();
            }

            if (ImGui::Button("Cancel")) {
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }

        if (ImGui::Button("Delete Sphere...")) {
            ImGui::OpenPopup("Delete Sphere");
        }

        // Always center this window when appearing
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

        if (ImGui::BeginPopupModal("Delete Sphere", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Enther the sphere ID you want to delete");
            ImGui::InputInt("Sphere ID", &m_SphereID);

            for (int i = 0; i < m_World->objects.size(); i++) {
                if (m_SphereID == i) {
                    if (ImGui::Button("Delete Sphere")) {
                        DeleteSphere(m_World->objects.at(m_SphereID));
                        m_World->objects.erase(m_World->objects.begin() + m_SphereID);
                        ImGui::CloseCurrentPopup();
                    }
                }
            }

            if (ImGui::Button("Cancel")) {
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }
    }

    ImGui::End();
}

bool CudaLayer::OnImGuiResize()
{
    ImVec2 view = ImGui::GetContentRegionAvail();

    if (view.x != m_ImageWidth || view.y != m_ImageHeight) {
        if (view.x == 0 || view.y == 0) {
            return false;
        }

        m_ImageWidth = view.x;
        m_ImageHeight = view.y;

        m_NumTexels = m_ImageWidth * m_ImageHeight;
        m_NumValues = m_NumTexels * 4;
        m_SizeTexData = sizeof(GLubyte) * m_NumValues;

        checkCudaErrors(cudaFree(m_CudaDevRenderBuffer));
        checkCudaErrors(cudaFree(m_DrandState));
        checkCudaErrors(cudaFree(m_DrandState2));
        checkCudaErrors(cudaGraphicsUnregisterResource(m_CudaTexResource));

        InitCudaBuffers();
        InitGLBuffers();
        RunCudaInit();

        return true;
    }
    return true;
}

void CudaLayer::InitCudaBuffers()
{
    checkCudaErrors(cudaMalloc(&m_CudaDevRenderBuffer, m_SizeTexData)); // Allocate CUDA memory for color output
    // Allocate random state
    checkCudaErrors(cudaMalloc((void **)&m_DrandState, m_NumTexels * sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void **)&m_DrandState2, 1 * sizeof(curandState)));
    // Allocate Hittables
    // checkCudaErrors(cudaMalloc((void **)&m_HittableList, m_NumHittables * sizeof(Sphere*)));
    // checkCudaErrors(cudaMalloc((void **)&m_World, sizeof(Sphere*)));
}

void CudaLayer::InitGLBuffers()
{
    // create an OpenGL texture
    glGenTextures(1, &m_Texture);              // generate 1 texture
    glBindTexture(GL_TEXTURE_2D, m_Texture); // set it as current target
    // set basic texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // Specify 2D texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_ImageWidth, m_ImageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    // Register this texture with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterImage(&m_CudaTexResource, m_Texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    // SDK_CHECK_ERROR_GL();
}

void CudaLayer::RunCudaInit()
{
    LaunchRandInit(m_DrandState2);

    dim3 block(16, 16, 1);
    dim3 grid(m_ImageWidth / block.x, m_ImageHeight / block.y, 1);

    LaunchRenderInit(grid, block, m_ImageWidth, m_ImageHeight, m_DrandState);

    // LaunchCreateWorld(m_HittableList, m_World, m_DrandState2);
}

void CudaLayer::GenerateWorld()
{
    // Sphere* new_sphere;
    // checkCudaErrors(cudaMallocManaged(&new_sphere, sizeof(Sphere)));
    // checkCudaErrors(cudaMallocManaged(&new_sphere->mat_ptr, sizeof(Material)));
    // m_World->Add(new(new_sphere) Sphere(Vec3(0, -1000.0f, -1), 1000.0f, new(new_sphere->mat_ptr) Material(Vec3(0.5f, 0.5f, 0.5f), Mat::lambertian)));

    // std::srand(std::time(nullptr));
    // for (int a = -2; a < 2; a++) {
    //     for (int b = -2; b < 2; b++) {
    //         float choose_mat = (std::rand() % 2) + 1;
    //         Vec3 center = Vec3(a + RND, 0.2, b + RND);
    //         if (choose_mat == 1) {
    //             Sphere *lamertian;
    //             checkCudaErrors(cudaMallocManaged(&lamertian, sizeof(Sphere)));
    //             checkCudaErrors(cudaMallocManaged(&lamertian->mat_ptr, sizeof(Material)));
    //             m_World->Add(new (lamertian) Sphere(center, 0.2, new(lamertian->mat_ptr) Material(Vec3(RND, RND, RND), Mat::lambertian)));
    //         }
    //         else if (choose_mat == 2) {
    //             Sphere *metal;
    //             checkCudaErrors(cudaMallocManaged(&metal, sizeof(Sphere)));
    //             checkCudaErrors(cudaMallocManaged(&metal->mat_ptr, sizeof(Material)));
    //             m_World->Add(new (metal) Sphere(center, 0.2, new(metal->mat_ptr) Material(Vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND, Mat::metal)));
    //         }
    //     }
    // }

    // checkCudaErrors(cudaMallocManaged(&new_sphere, sizeof(Sphere)));
    // checkCudaErrors(cudaMallocManaged(&new_sphere->mat_ptr, sizeof(Material)));
    // m_World->Add(new(new_sphere) Sphere(Vec3(-4, 1, 0), 1.0f, new(new_sphere->mat_ptr) Material(Vec3(0.4f, 0.2f, 0.1f), Mat::lambertian)));

    // Texture* checker = new Texture(new Texture(Vec3(0.2f, 0.3f, 0.1f), Tex::constant_texture), new Texture(Vec3(0.9f, 0.9f, 0.9f), Tex::constant_texture), Tex::checker_texture);
    // Texture* color1 = new Texture(Vec3(1.0f, 0.0f, 0.0f), Tex::constant_texture);
    // checkCudaErrors(cudaMallocManaged(&color1, sizeof(Texture)));

    Sphere* groundSphere;
    checkCudaErrors(cudaMallocManaged(&groundSphere, sizeof(Sphere)));
    checkCudaErrors(cudaMallocManaged(&groundSphere->mat_ptr, sizeof(Material)));
    checkCudaErrors(cudaMallocManaged(&groundSphere->mat_ptr->albedo, sizeof(Texture)));
    checkCudaErrors(cudaMallocManaged(&groundSphere->mat_ptr->albedo->odd, sizeof(Texture)));
    checkCudaErrors(cudaMallocManaged(&groundSphere->mat_ptr->albedo->even, sizeof(Texture)));
    m_World->Add(new(groundSphere) Sphere(Vec3(0.0f, -1000.5f, 0.0f), 1000.0f, new(groundSphere->mat_ptr) Material(new(groundSphere->mat_ptr->albedo) Texture(new(groundSphere->mat_ptr->albedo->odd) Texture(Vec3(0.2f, 0.3f, 0.1f), Tex::constant_texture), new(groundSphere->mat_ptr->albedo->even) Texture(Vec3(0.9f, 0.9f, 0.9f), Tex::constant_texture), Tex::checker_texture), Mat::lambertian)));

    int width, height, nr;
    unsigned char* data;
    unsigned char* odata;
    const char* filename = "assets/textures/earrrth.jpeg";

    data = stbi_load(filename, &width, &height, &nr, 0);

    if (!data) {
        RT_ERROR("ERROR: Could not load texture image file {0}", filename);
        width = height = 0;
    }

    Sphere* earth_sphere;
    checkCudaErrors(cudaMallocManaged(&earth_sphere, sizeof(Sphere)));
    checkCudaErrors(cudaMallocManaged(&earth_sphere->mat_ptr, sizeof(Material)));
    checkCudaErrors(cudaMallocManaged(&earth_sphere->mat_ptr->albedo, sizeof(Texture)));
    checkCudaErrors(cudaMallocManaged(&odata, width * height * nr * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(odata, data, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    m_World->Add(new(earth_sphere) Sphere(Vec3(0.0f, 3.5f, -1.0f), 2.0f, new(earth_sphere->mat_ptr) Material(new(earth_sphere->mat_ptr->albedo) Texture(odata, width, height, Tex::image_texture), Mat::lambertian)));

    Sphere* sphere1;
    checkCudaErrors(cudaMallocManaged(&sphere1, sizeof(Sphere)));
    checkCudaErrors(cudaMallocManaged(&sphere1->mat_ptr, sizeof(Material)));
    checkCudaErrors(cudaMallocManaged(&sphere1->mat_ptr->albedo, sizeof(Texture)));
    m_World->Add(new(sphere1) Sphere(Vec3(0.0f, 0.0f, -1.0f), 0.5f, new(sphere1->mat_ptr) Material(new(sphere1->mat_ptr->albedo) Texture(Vec3(0.1f, 0.2f, 0.5f), Tex::constant_texture), Mat::lambertian)));

    Sphere* sphere2;
    checkCudaErrors(cudaMallocManaged(&sphere2, sizeof(Sphere)));
    checkCudaErrors(cudaMallocManaged(&sphere2->mat_ptr, sizeof(Material)));
    checkCudaErrors(cudaMallocManaged(&sphere2->mat_ptr->albedo, sizeof(Texture)));
    m_World->Add(new(sphere2) Sphere(Vec3(1.0f, 0.0f, -1.0f), 0.5f, new(sphere2->mat_ptr) Material(new(sphere2->mat_ptr->albedo) Texture(Vec3(0.8f, 0.6f, 0.2f), Tex::constant_texture), 0.0f, Mat::metal)));

    Sphere* glassSphere_a;
    checkCudaErrors(cudaMallocManaged(&glassSphere_a, sizeof(Sphere)));
    checkCudaErrors(cudaMallocManaged(&glassSphere_a->mat_ptr, sizeof(Material)));
    m_World->Add(new(glassSphere_a) Sphere(Vec3(-1.0f, 0.0f, -1.0f), 0.5f, new(glassSphere_a->mat_ptr) Material(1.5f, Mat::dielectric)));

    Sphere* glassSphere_b;
    checkCudaErrors(cudaMallocManaged(&glassSphere_b, sizeof(Sphere)));
    checkCudaErrors(cudaMallocManaged(&glassSphere_b->mat_ptr, sizeof(Material)));
    m_World->Add(new(glassSphere_b) Sphere(Vec3(-1.0f, 0.0f, -1.0f), -0.45f, new(glassSphere_b->mat_ptr) Material(1.5f, Mat::dielectric)));
}

void CudaLayer::AddSphere()
{
    Sphere* new_sphere;
    checkCudaErrors(cudaMallocManaged(&new_sphere, sizeof(Sphere)));
    checkCudaErrors(cudaMallocManaged(&new_sphere->mat_ptr, sizeof(Material)));
    checkCudaErrors(cudaMallocManaged(&new_sphere->mat_ptr->albedo, sizeof(Texture)));
    if (m_UseLambertian) {
        if (m_UseConstantTexture) {
            m_World->Add(new(new_sphere) Sphere(m_SpherePosition, m_SphereRadius, new(new_sphere->mat_ptr) Material(new(new_sphere->mat_ptr->albedo) Texture(m_newColor, Tex::constant_texture), Mat::lambertian)));
        }
        else {
            checkCudaErrors(cudaMallocManaged(&new_sphere->mat_ptr->albedo->even, sizeof(Texture)));
            checkCudaErrors(cudaMallocManaged(&new_sphere->mat_ptr->albedo->odd, sizeof(Texture)));
            m_World->Add(new(new_sphere) Sphere(m_SpherePosition, m_SphereRadius, new(new_sphere->mat_ptr) Material(new(new_sphere->mat_ptr->albedo) Texture(new(new_sphere->mat_ptr->albedo->odd) Texture(Vec3(0.2f, 0.3f, 0.1f), Tex::constant_texture), new(new_sphere->mat_ptr->albedo->even) Texture(Vec3(0.9f, 0.9f, 0.9f), Tex::constant_texture), Tex::checker_texture), Mat::lambertian)));
        }
    }
    else if (m_UseMetal) {
        if (m_UseConstantTexture) {
            m_World->Add(new(new_sphere) Sphere(m_SpherePosition, m_SphereRadius, new(new_sphere->mat_ptr) Material(new(new_sphere->mat_ptr->albedo) Texture(m_newColor, Tex::constant_texture), m_Fuzz, Mat::metal)));
        }
        else {
            checkCudaErrors(cudaMallocManaged(&new_sphere->mat_ptr->albedo->even, sizeof(Texture)));
            checkCudaErrors(cudaMallocManaged(&new_sphere->mat_ptr->albedo->odd, sizeof(Texture)));
            m_World->Add(new(new_sphere) Sphere(m_SpherePosition, m_SphereRadius, new(new_sphere->mat_ptr) Material(new(new_sphere->mat_ptr->albedo) Texture(new(new_sphere->mat_ptr->albedo->odd) Texture(Vec3(0.2f, 0.3f, 0.1f), Tex::constant_texture), new(new_sphere->mat_ptr->albedo->even) Texture(Vec3(0.9f, 0.9f, 0.9f), Tex::constant_texture), Tex::checker_texture), m_Fuzz, Mat::metal)));
        }
    }
    else {
        m_World->Add(new(new_sphere) Sphere(m_SpherePosition, m_SphereRadius, new(new_sphere->mat_ptr) Material(m_IR, Mat::dielectric)));
    }
}

void CudaLayer::DeleteSphere(Sphere* sphere)
{
    if (sphere->mat_ptr->albedo != nullptr) {
        checkCudaErrors(cudaFree(sphere->mat_ptr->albedo->odd));
        checkCudaErrors(cudaFree(sphere->mat_ptr->albedo->even));
        checkCudaErrors(cudaFree(sphere->mat_ptr->albedo->data));
    }
    checkCudaErrors(cudaFree(sphere->mat_ptr->albedo));
    checkCudaErrors(cudaFree(sphere->mat_ptr));
    checkCudaErrors(cudaFree(sphere));
}

void CudaLayer::RunCudaUpdate()
{
    LaunchKernel(static_cast<unsigned int *>(m_CudaDevRenderBuffer), m_ImageWidth, m_ImageHeight, m_SamplesPerPixel, m_MaxDepth, m_World, m_DrandState, m_Inputs);

    // We want to copy cuda_dev_render_buffer data to the texture.
    // Map buffer objects to get CUDA device pointers.
    cudaArray *texture_ptr;
    checkCudaErrors(cudaGraphicsMapResources(1, &m_CudaTexResource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, m_CudaTexResource, 0, 0));

    checkCudaErrors(cudaMemcpy2DToArray(texture_ptr, 0, 0, m_CudaDevRenderBuffer, m_ImageWidth * sizeof(unsigned int), m_ImageWidth * sizeof(unsigned int), m_ImageHeight, cudaMemcpyDeviceToDevice));
    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, &m_CudaTexResource, 0));
}