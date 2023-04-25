#include "Cuda/CudaLayer.h"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

#include "../Utils/RawStbImage.h"

#include "ImGui/ImGuiFileDialog.h"

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
        DeleteHittable(obj);
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

    ImGui::Text("Generated image dimensions: %dx%d", m_ImageWidth, m_ImageHeight);

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

    ImGui::Separator();

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

    ImGui::Separator();

    if (ImGui::CollapsingHeader("Ray Tracing Settings", base_flags)) {
        ImGui::SliderInt("Samples Per Pixel", (int *)&m_SamplesPerPixel, 1, 1000);
        ImGui::SliderInt("Max Depth", (int *)&m_MaxDepth, 1, 50);
    }

    ImGui::Separator();

    if (ImGui::CollapsingHeader("Hittables Settings", base_flags)) {
        for (int i = 0; i < m_World->objects.size(); i++) {
            if (ImGui::TreeNodeEx((GetTextForEnum(m_World->objects.at(i)->hittable) + std::to_string(i)).c_str())) {

                if (m_World->objects.at(i)->hittable == Hitt::sphere) {
                    ImGui::DragFloat3("Position", (float *)&m_World->objects.at(i)->center, 0.01f, -FLT_MAX, FLT_MAX, "%.2f");
                    ImGui::DragFloat("Radius", (float *)&m_World->objects.at(i)->radius, 0.01f, -FLT_MAX, FLT_MAX, "%.2f");
                }
                else {
                    ImGui::DragFloat3("Position", (float *)&m_World->objects.at(i)->center, 0.01f, -FLT_MAX, FLT_MAX, "%.2f");
                    ImGui::DragFloat("Width", (float *)&m_World->objects.at(i)->width, 0.01f, 0, FLT_MAX, "%.2f");
                    ImGui::DragFloat("Height", (float *)&m_World->objects.at(i)->height, 0.01f, 0, FLT_MAX, "%.2f");
                }

                if (ImGui::TreeNodeEx("Material", base_flags)) {
                    const char* mat_items[] = {"Lambertian", "Metal", "Dielectric", "Diffuse Light"};
                    int mat_item_current = m_World->objects.at(i)->mat_ptr->material;

                    if (ImGui::Combo(" ", &mat_item_current, mat_items, IM_ARRAYSIZE(mat_items))) {
                        m_World->objects.at(i)->mat_ptr->material = (Mat)mat_item_current;
                    }

                    if (m_World->objects.at(i)->mat_ptr->material == Mat::metal) {
                        ImGui::DragFloat(("Fuzziness " + std::to_string(i)).c_str(), (float *)&m_World->objects.at(i)->mat_ptr->fuzz, 0.01f, 0.0f, 1.0f, "%.2f");
                    }

                    if (m_World->objects.at(i)->mat_ptr->material == Mat::diffuse_light) {
                        ImGui::SliderInt("Light Intensity", &m_World->objects.at(i)->mat_ptr->light_intensity, 0, 10);
                    }

                    if (m_World->objects.at(i)->mat_ptr->material != Mat::dielectric) {
                        if (ImGui::TreeNodeEx("Texture", base_flags)) {
                            const char* tex_items[] = {"Constant", "Checker", "Image"};
                            int tex_item_current = m_World->objects.at(i)->mat_ptr->albedo->texture;

                            if (ImGui::Combo(" ", &tex_item_current, tex_items, IM_ARRAYSIZE(tex_items))) {
                                m_World->objects.at(i)->mat_ptr->albedo->texture = (Tex)tex_item_current;
                            }

                            if (m_World->objects.at(i)->mat_ptr->albedo->texture == Tex::constant_texture) {
                                ImGui::ColorEdit3(("Albedo " + std::to_string(i)).c_str(), (float *)&m_World->objects.at(i)->mat_ptr->albedo->color);
                            }
                            else if (m_World->objects.at(i)->mat_ptr->albedo->texture == Tex::checker_texture) {
                                ImGui::ColorEdit3(("Albedo odd " + std::to_string(i)).c_str(), (float *)&m_World->objects.at(i)->mat_ptr->albedo->odd->color);
                                ImGui::ColorEdit3(("Albedo even " + std::to_string(i)).c_str(), (float *)&m_World->objects.at(i)->mat_ptr->albedo->even->color);
                            }
                            else if (m_World->objects.at(i)->mat_ptr->albedo->texture == Tex::image_texture) {

                                if (ImGui::Button("Open...")) {
                                    m_ButtonID = i;
                                    ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".jpg,.jpeg,.png", ".", 1, nullptr, ImGuiFileDialogFlags_Modal);
                                }

                                // Always center this window when appearing
                                ImVec2 center = ImGui::GetMainViewport()->GetCenter();
                                ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

                                // display
                                if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey")) {
                                    // action if OK
                                    if (ImGuiFileDialog::Instance()->IsOk()) {

                                        m_TextureImageFilename = ("assets/textures/" + ImGuiFileDialog::Instance()->GetCurrentFileName()).c_str();

                                        if (m_World->objects.at(m_ButtonID)->mat_ptr->albedo->data != nullptr) {
                                            checkCudaErrors(cudaFree(m_World->objects.at(m_ButtonID)->mat_ptr->albedo->data));
                                        }

                                        m_TextureImageData = LoadImage(m_TextureImageFilename, m_TextureImageData, &m_TextureImageWidth, &m_TextureImageHeight, &m_TextureImageNR);
                                        checkCudaErrors(cudaMallocManaged(&m_World->objects.at(m_ButtonID)->mat_ptr->albedo->data, m_TextureImageWidth * m_TextureImageHeight * m_TextureImageNR * sizeof(unsigned char)));
                                        checkCudaErrors(cudaMemcpy(m_World->objects.at(m_ButtonID)->mat_ptr->albedo->data, m_TextureImageData, m_TextureImageWidth * m_TextureImageHeight * m_TextureImageNR * sizeof(unsigned char), cudaMemcpyHostToDevice));
                                        STBI_FREE(m_TextureImageData);

                                        m_World->objects.at(m_ButtonID)->mat_ptr->albedo->width = m_TextureImageWidth; 
                                        m_World->objects.at(m_ButtonID)->mat_ptr->albedo->height = m_TextureImageHeight; 

                                        if (m_World->objects.at(m_ButtonID)->hittable == Hitt::sphere) {
                                            m_World->objects.at(m_ButtonID) = new(m_World->objects.at(m_ButtonID)) Hittable(m_World->objects.at(m_ButtonID)->center, m_World->objects.at(m_ButtonID)->radius, new(m_World->objects.at(m_ButtonID)->mat_ptr) Material(new(m_World->objects.at(m_ButtonID)->mat_ptr->albedo) Texture(m_World->objects.at(m_ButtonID)->mat_ptr->albedo->data, m_TextureImageWidth, m_TextureImageHeight, Tex::image_texture), m_World->objects.at(m_ButtonID)->mat_ptr->material), Hitt::sphere);
                                        }
                                        else if (m_World->objects.at(m_ButtonID)->hittable == Hitt::xy_rect) {
                                            m_World->objects.at(m_ButtonID) = new(m_World->objects.at(m_ButtonID)) Hittable(m_World->objects.at(m_ButtonID)->center, m_World->objects.at(m_ButtonID)->width, m_World->objects.at(m_ButtonID)->height, new(m_World->objects.at(m_ButtonID)->mat_ptr) Material(new(m_World->objects.at(m_ButtonID)->mat_ptr->albedo) Texture(m_World->objects.at(m_ButtonID)->mat_ptr->albedo->data, m_TextureImageWidth, m_TextureImageHeight, Tex::image_texture), m_World->objects.at(m_ButtonID)->mat_ptr->material), Hitt::xy_rect);
                                        }
                                        else if (m_World->objects.at(m_ButtonID)->hittable == Hitt::xz_rect) {
                                            m_World->objects.at(m_ButtonID) = new(m_World->objects.at(m_ButtonID)) Hittable(m_World->objects.at(m_ButtonID)->center, m_World->objects.at(m_ButtonID)->width, m_World->objects.at(m_ButtonID)->height, new(m_World->objects.at(m_ButtonID)->mat_ptr) Material(new(m_World->objects.at(m_ButtonID)->mat_ptr->albedo) Texture(m_World->objects.at(m_ButtonID)->mat_ptr->albedo->data, m_TextureImageWidth, m_TextureImageHeight, Tex::image_texture), m_World->objects.at(m_ButtonID)->mat_ptr->material), Hitt::xz_rect);
                                        }
                                        else if (m_World->objects.at(m_ButtonID)->hittable == Hitt::yz_rect) {
                                            m_World->objects.at(m_ButtonID) = new(m_World->objects.at(m_ButtonID)) Hittable(m_World->objects.at(m_ButtonID)->center, m_World->objects.at(m_ButtonID)->width, m_World->objects.at(m_ButtonID)->height, new(m_World->objects.at(m_ButtonID)->mat_ptr) Material(new(m_World->objects.at(m_ButtonID)->mat_ptr->albedo) Texture(m_World->objects.at(m_ButtonID)->mat_ptr->albedo->data, m_TextureImageWidth, m_TextureImageHeight, Tex::image_texture), m_World->objects.at(m_ButtonID)->mat_ptr->material), Hitt::yz_rect);
                                        }

                                        // don't forget to set the path for the object
                                        m_World->objects.at(m_ButtonID)->mat_ptr->albedo->path = m_TextureImageFilename;
                                    }

                                    // close
                                    ImGuiFileDialog::Instance()->Close();
                                }

                                if (m_World->objects.at(i)->mat_ptr->albedo->path != nullptr) {
                                    ImGui::Text(m_World->objects.at(i)->mat_ptr->albedo->path);
                                }
                                else {
                                    ImGui::Text("None");
                                }
                            }

                            ImGui::TreePop();
                        }
                    }
                    else if (m_World->objects.at(i)->mat_ptr->material == Mat::dielectric) {
                        ImGui::DragFloat(("Index of Refraction " + std::to_string(i)).c_str(), (float *)&m_World->objects.at(i)->mat_ptr->ir, 0.01f, 0.0f, FLT_MAX, "%.2f");
                    }

                    ImGui::TreePop();
                }

                ImGui::TreePop();
            }
        }

        if (ImGui::Button("Add Hittable...")) {
            ImGui::OpenPopup("New Hittable");
        }

        // Always center this window when appearing
        ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

        if (ImGui::BeginPopupModal("New Hittable", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {

            ImGui::Separator();
            ImGui::Text("Choose the type of hittable:");
            ImGui::Separator();

            if (ImGui::Checkbox("Sphere", &m_UseHittableSphere)) {
                m_UseHittableXYRect = false;
                m_UseHittableXZRect = false;
                m_UseHittableYZRect = false;
            }
            else if (ImGui::Checkbox("XYRect", &m_UseHittableXYRect)) {
                m_UseHittableSphere = false;
                m_UseHittableXZRect = false;
                m_UseHittableYZRect = false;
            }
            else if (ImGui::Checkbox("XZRect", &m_UseHittableXZRect)) {
                m_UseHittableSphere = false;
                m_UseHittableXYRect = false;
                m_UseHittableYZRect = false;
            }
            else if (ImGui::Checkbox("YZRect", &m_UseHittableYZRect)) {
                m_UseHittableSphere = false;
                m_UseHittableXYRect = false;
                m_UseHittableXZRect = false;
            }

            ImGui::Separator();
            ImGui::Text("Choose the hittable material:");
            ImGui::Separator();

            if (ImGui::Checkbox("Lambertian", &m_UseLambertian)) {
                m_UseMetal = false;
                m_UseDielectric = false;
                m_UseDiffuseLight = false;
            }
            else if (ImGui::Checkbox("Metal", &m_UseMetal)) {
                m_UseLambertian = false;
                m_UseDielectric = false;
                m_UseDiffuseLight = false;
            }
            else if (ImGui::Checkbox("Dielectric", &m_UseDielectric)) {
                m_UseMetal = false;
                m_UseLambertian = false;
                m_UseDiffuseLight = false;
            }
            else if (ImGui::Checkbox("Diffuse Light", &m_UseDiffuseLight)) {
                m_UseMetal = false;
                m_UseLambertian = false;
                m_UseDielectric = false;
            }

            ImGui::Separator();

            if (m_UseLambertian == true || m_UseMetal == true || m_UseDiffuseLight == true) {
                ImGui::Text("Choose the hittable material texture:");
                ImGui::Separator();
                if (ImGui::Checkbox("Constant Texture", &m_UseConstantTexture)) {
                    m_UseCheckerTexture = false;
                    m_UseImageTexture = false;
                }
                else if (ImGui::Checkbox("Checker Texture", &m_UseCheckerTexture)) {
                    m_UseConstantTexture = false;
                    m_UseImageTexture = false;
                }
                else if (ImGui::Checkbox("Image Texture", &m_UseImageTexture)) {
                    m_UseConstantTexture = false;
                    m_UseCheckerTexture = false;
                }
                ImGui::Separator();
            }

            if (((m_UseLambertian || m_UseMetal || m_UseDielectric || m_UseDiffuseLight) && (m_UseHittableSphere || m_UseHittableXYRect || m_UseHittableXZRect || m_UseHittableYZRect))) {
                if (!m_UseDielectric) {
                    if (m_UseConstantTexture || m_UseCheckerTexture || m_UseImageTexture) {
                        if (ImGui::Button("Add")) {
                            AddHittable();
                            ImGui::CloseCurrentPopup();
                        }
                    }
                }
                else {
                    if (ImGui::Button("Add")) {
                        AddHittable();
                        ImGui::CloseCurrentPopup();
                    }
                }
            }

            if (ImGui::Button("Cancel")) {
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }

        if (ImGui::Button("Delete Hittable...")) {
            ImGui::OpenPopup("Delete Hittable");
        }

        // Always center this window when appearing
        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

        if (ImGui::BeginPopupModal("Delete Hittable", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Enter the hittable ID you want to delete");
            ImGui::InputInt("Hittable ID", &m_HittableID);

            for (int i = 0; i < m_World->objects.size(); i++) {
                if (m_HittableID == i) {
                    if (ImGui::Button("Delete")) {
                        DeleteHittable(m_World->objects.at(m_HittableID));
                        m_World->objects.erase(m_World->objects.begin() + m_HittableID);
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

    ImGui::Separator();

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
    Hittable* ground;
    checkCudaErrors(cudaMallocManaged(&ground, sizeof(Hittable)));
    checkCudaErrors(cudaMallocManaged(&ground->mat_ptr, sizeof(Material)));
    checkCudaErrors(cudaMallocManaged(&ground->mat_ptr->albedo, sizeof(Texture)));
    checkCudaErrors(cudaMallocManaged(&ground->mat_ptr->albedo->odd, sizeof(Texture)));
    checkCudaErrors(cudaMallocManaged(&ground->mat_ptr->albedo->even, sizeof(Texture)));
    m_World->Add(new(ground) Hittable(Vec3(0.0f, -0.5f, 0.0f), 1000.0f, 1000.0f, new(ground->mat_ptr) Material(new(ground->mat_ptr->albedo) Texture(new(ground->mat_ptr->albedo->odd) Texture(Vec3(0.2f, 0.3f, 0.1f), Tex::constant_texture), new(ground->mat_ptr->albedo->even) Texture(Vec3(0.9f, 0.9f, 0.9f), Tex::constant_texture), Tex::checker_texture), Mat::lambertian), Hitt::xz_rect));

    Hittable* skybox_sphere;
    checkCudaErrors(cudaMallocManaged(&skybox_sphere, sizeof(Hittable)));
    checkCudaErrors(cudaMallocManaged(&skybox_sphere->mat_ptr, sizeof(Material)));
    checkCudaErrors(cudaMallocManaged(&skybox_sphere->mat_ptr->albedo, sizeof(Texture)));
    checkCudaErrors(cudaMallocManaged(&skybox_sphere->mat_ptr->albedo->odd, sizeof(Texture)));
    checkCudaErrors(cudaMallocManaged(&skybox_sphere->mat_ptr->albedo->even, sizeof(Texture)));
    m_TextureImageFilename = "assets/textures/industrial_sunset_puresky.jpg";
    m_TextureImageData = LoadImage(m_TextureImageFilename, m_TextureImageData, &m_TextureImageWidth, &m_TextureImageHeight, &m_TextureImageNR);
    checkCudaErrors(cudaMallocManaged(&skybox_sphere->mat_ptr->albedo->data, m_TextureImageWidth * m_TextureImageHeight * m_TextureImageNR * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(skybox_sphere->mat_ptr->albedo->data, m_TextureImageData, m_TextureImageWidth * m_TextureImageHeight * m_TextureImageNR * sizeof(unsigned char), cudaMemcpyHostToDevice));
    STBI_FREE(m_TextureImageData);
    m_World->Add(new(skybox_sphere) Hittable(Vec3(0.0f, 0.0f, 0.0f), 1000.0f, new(skybox_sphere->mat_ptr) Material(new(skybox_sphere->mat_ptr->albedo) Texture(skybox_sphere->mat_ptr->albedo->data, m_TextureImageWidth, m_TextureImageHeight, Tex::image_texture), Mat::lambertian), Hitt::sphere));
    // don't forget to set the path for the object
    skybox_sphere->mat_ptr->albedo->path = m_TextureImageFilename;

    Hittable* sphere1;
    checkCudaErrors(cudaMallocManaged(&sphere1, sizeof(Hittable)));
    checkCudaErrors(cudaMallocManaged(&sphere1->mat_ptr, sizeof(Material)));
    checkCudaErrors(cudaMallocManaged(&sphere1->mat_ptr->albedo, sizeof(Texture)));
    checkCudaErrors(cudaMallocManaged(&sphere1->mat_ptr->albedo->odd, sizeof(Texture)));
    checkCudaErrors(cudaMallocManaged(&sphere1->mat_ptr->albedo->even, sizeof(Texture)));
    m_World->Add(new(sphere1) Hittable(Vec3(0.0f, 0.0f, -1.0f), 0.5f, new(sphere1->mat_ptr) Material(new(sphere1->mat_ptr->albedo) Texture(Vec3(0.1f, 0.2f, 0.5f), Tex::constant_texture), Mat::lambertian), Hitt::sphere));

    Hittable* sphere2;
    checkCudaErrors(cudaMallocManaged(&sphere2, sizeof(Hittable)));
    checkCudaErrors(cudaMallocManaged(&sphere2->mat_ptr, sizeof(Material)));
    checkCudaErrors(cudaMallocManaged(&sphere2->mat_ptr->albedo, sizeof(Texture)));
    checkCudaErrors(cudaMallocManaged(&sphere2->mat_ptr->albedo->odd, sizeof(Texture)));
    checkCudaErrors(cudaMallocManaged(&sphere2->mat_ptr->albedo->even, sizeof(Texture)));
    m_World->Add(new(sphere2) Hittable(Vec3(1.0f, 0.0f, -1.0f), 0.5f, new(sphere2->mat_ptr) Material(new(sphere2->mat_ptr->albedo) Texture(Vec3(0.8f, 0.6f, 0.2f), Tex::constant_texture), 0.0f, Mat::metal), Hitt::sphere));

    Hittable* glassSphere_a;
    checkCudaErrors(cudaMallocManaged(&glassSphere_a, sizeof(Hittable)));
    checkCudaErrors(cudaMallocManaged(&glassSphere_a->mat_ptr, sizeof(Material)));
    checkCudaErrors(cudaMallocManaged(&glassSphere_a->mat_ptr->albedo, sizeof(Texture)));
    checkCudaErrors(cudaMallocManaged(&glassSphere_a->mat_ptr->albedo->odd, sizeof(Texture)));
    checkCudaErrors(cudaMallocManaged(&glassSphere_a->mat_ptr->albedo->even, sizeof(Texture)));
    m_World->Add(new(glassSphere_a) Hittable(Vec3(-1.0f, 0.0f, -1.0f), 0.5f, new(glassSphere_a->mat_ptr) Material(1.5f, Mat::dielectric), Hitt::sphere));

    Hittable* glassSphere_b;
    checkCudaErrors(cudaMallocManaged(&glassSphere_b, sizeof(Hittable)));
    checkCudaErrors(cudaMallocManaged(&glassSphere_b->mat_ptr, sizeof(Material)));
    checkCudaErrors(cudaMallocManaged(&glassSphere_b->mat_ptr->albedo, sizeof(Texture)));
    checkCudaErrors(cudaMallocManaged(&glassSphere_b->mat_ptr->albedo->odd, sizeof(Texture)));
    checkCudaErrors(cudaMallocManaged(&glassSphere_b->mat_ptr->albedo->even, sizeof(Texture)));
    m_World->Add(new(glassSphere_b) Hittable(Vec3(-1.0f, 0.0f, -1.0f), -0.45f, new(glassSphere_b->mat_ptr) Material(1.5f, Mat::dielectric), Hitt::sphere));

    Hittable* light_sphere;
    checkCudaErrors(cudaMallocManaged(&light_sphere, sizeof(Hittable)));
    checkCudaErrors(cudaMallocManaged(&light_sphere->mat_ptr, sizeof(Material)));
    checkCudaErrors(cudaMallocManaged(&light_sphere->mat_ptr->albedo, sizeof(Texture)));
    checkCudaErrors(cudaMallocManaged(&light_sphere->mat_ptr->albedo->odd, sizeof(Texture)));
    checkCudaErrors(cudaMallocManaged(&light_sphere->mat_ptr->albedo->even, sizeof(Texture)));
    m_TextureImageFilename = "assets/textures/8k_sun.jpg";
    m_TextureImageData = LoadImage(m_TextureImageFilename, m_TextureImageData, &m_TextureImageWidth, &m_TextureImageHeight, &m_TextureImageNR);
    checkCudaErrors(cudaMallocManaged(&light_sphere->mat_ptr->albedo->data, m_TextureImageWidth * m_TextureImageHeight * m_TextureImageNR * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(light_sphere->mat_ptr->albedo->data, m_TextureImageData, m_TextureImageWidth * m_TextureImageHeight * m_TextureImageNR * sizeof(unsigned char), cudaMemcpyHostToDevice));
    STBI_FREE(m_TextureImageData);
    m_World->Add(new(light_sphere) Hittable(Vec3(0.0f, 2.0f, 0.0f), 0.5f, new(light_sphere->mat_ptr) Material(new(light_sphere->mat_ptr->albedo) Texture(light_sphere->mat_ptr->albedo->data, m_TextureImageWidth, m_TextureImageHeight, Tex::image_texture), m_LightIntensity, Mat::diffuse_light), Hitt::sphere));
    // don't forget to set the path for the object
    light_sphere->mat_ptr->albedo->path = m_TextureImageFilename;

    Hittable* rect;
    checkCudaErrors(cudaMallocManaged(&rect, sizeof(Hittable)));
    checkCudaErrors(cudaMallocManaged(&rect->mat_ptr, sizeof(Material)));
    checkCudaErrors(cudaMallocManaged(&rect->mat_ptr->albedo, sizeof(Texture)));
    checkCudaErrors(cudaMallocManaged(&rect->mat_ptr->albedo->odd, sizeof(Texture)));
    checkCudaErrors(cudaMallocManaged(&rect->mat_ptr->albedo->even, sizeof(Texture)));
    m_World->Add(new(rect) Hittable(Vec3(0.0f, 1.0f, -3.0f), 6.0f, 3.0f, new(rect->mat_ptr) Material(new(rect->mat_ptr->albedo) Texture(Vec3(1.0f, 0.0f, 0.0f), Tex::constant_texture), 7, Mat::diffuse_light), Hitt::xy_rect));
}

void CudaLayer::AddHittable()
{
    Hittable* new_hittable;
    checkCudaErrors(cudaMallocManaged(&new_hittable, sizeof(Hittable)));
    checkCudaErrors(cudaMallocManaged(&new_hittable->mat_ptr, sizeof(Material)));
    checkCudaErrors(cudaMallocManaged(&new_hittable->mat_ptr->albedo, sizeof(Texture)));

    // FIXME: works for now... not a good solution
    new_hittable->mat_ptr->albedo->data = nullptr;

    checkCudaErrors(cudaMallocManaged(&new_hittable->mat_ptr->albedo->odd, sizeof(Texture)));
    checkCudaErrors(cudaMallocManaged(&new_hittable->mat_ptr->albedo->even, sizeof(Texture)));

    if (m_UseHittableSphere) {
        if (m_UseLambertian) {
            if (m_UseConstantTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_SphereRadius, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(m_newColor, Tex::constant_texture), Mat::lambertian), Hitt::sphere));
            }
            else if (m_UseCheckerTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_SphereRadius, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new(new_hittable->mat_ptr->albedo->odd) Texture(Vec3(0.0f, 0.0f, 0.0f), Tex::constant_texture), new(new_hittable->mat_ptr->albedo->even) Texture(Vec3(1.0f, 1.0f, 1.0f), Tex::constant_texture), Tex::checker_texture), Mat::lambertian), Hitt::sphere));
            }
            else if (m_UseImageTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_SphereRadius, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new_hittable->mat_ptr->albedo->data, m_TextureImageWidth, m_TextureImageHeight, Tex::image_texture), Mat::lambertian), Hitt::sphere));
            }
        }
        else if (m_UseMetal) {
            if (m_UseConstantTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_SphereRadius, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(m_newColor, Tex::constant_texture), m_Fuzz, Mat::metal), Hitt::sphere));
            }
            else if (m_UseCheckerTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_SphereRadius, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new(new_hittable->mat_ptr->albedo->odd) Texture(Vec3(0.0f, 0.0f, 0.0f), Tex::constant_texture), new(new_hittable->mat_ptr->albedo->even) Texture(Vec3(1.0f, 1.0f, 1.0f), Tex::constant_texture), Tex::checker_texture), m_Fuzz, Mat::metal), Hitt::sphere));
            }
            else if (m_UseImageTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_SphereRadius, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new_hittable->mat_ptr->albedo->data, m_TextureImageWidth, m_TextureImageHeight, Tex::image_texture), m_Fuzz, Mat::metal), Hitt::sphere));
            }
        }
        else if (m_UseDiffuseLight) {
            if (m_UseConstantTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_SphereRadius, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(m_newColor, Tex::constant_texture), m_LightIntensity, Mat::diffuse_light), Hitt::sphere));
            }
            else if (m_UseCheckerTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_SphereRadius, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new(new_hittable->mat_ptr->albedo->odd) Texture(Vec3(0.0f, 0.0f, 0.0f), Tex::constant_texture), new(new_hittable->mat_ptr->albedo->even) Texture(Vec3(1.0f, 1.0f, 1.0f), Tex::constant_texture), Tex::checker_texture), m_LightIntensity, Mat::diffuse_light), Hitt::sphere));
            }
            else if (m_UseImageTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_SphereRadius, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new_hittable->mat_ptr->albedo->data, m_TextureImageWidth, m_TextureImageHeight, Tex::image_texture), m_LightIntensity, Mat::diffuse_light), Hitt::sphere));
            }
        }
        else if (m_UseDielectric) {
            m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_SphereRadius, new(new_hittable->mat_ptr) Material(m_IR, Mat::dielectric), Hitt::sphere));
        }
    }
    else if (m_UseHittableXYRect) {
        if (m_UseLambertian) {
            if (m_UseConstantTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(m_newColor, Tex::constant_texture), Mat::lambertian), Hitt::xy_rect));
            }
            else if (m_UseCheckerTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new(new_hittable->mat_ptr->albedo->odd) Texture(Vec3(0.0f, 0.0f, 0.0f), Tex::constant_texture), new(new_hittable->mat_ptr->albedo->even) Texture(Vec3(1.0f, 1.0f, 1.0f), Tex::constant_texture), Tex::checker_texture), Mat::lambertian), Hitt::xy_rect));
            }
            else if (m_UseImageTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new_hittable->mat_ptr->albedo->data, m_TextureImageWidth, m_TextureImageHeight, Tex::image_texture), Mat::lambertian), Hitt::xy_rect));
            }
        }
        else if (m_UseMetal) {
            if (m_UseConstantTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(m_newColor, Tex::constant_texture), m_Fuzz, Mat::metal), Hitt::xy_rect));
            }
            else if (m_UseCheckerTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new(new_hittable->mat_ptr->albedo->odd) Texture(Vec3(0.0f, 0.0f, 0.0f), Tex::constant_texture), new(new_hittable->mat_ptr->albedo->even) Texture(Vec3(1.0f, 1.0f, 1.0f), Tex::constant_texture), Tex::checker_texture), m_Fuzz, Mat::metal), Hitt::xy_rect));
            }
            else if (m_UseImageTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new_hittable->mat_ptr->albedo->data, m_TextureImageWidth, m_TextureImageHeight, Tex::image_texture), m_Fuzz, Mat::metal), Hitt::xy_rect));
            }
        }
        else if (m_UseDiffuseLight) {
            if (m_UseConstantTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(m_newColor, Tex::constant_texture), m_LightIntensity, Mat::diffuse_light), Hitt::xy_rect));
            }
            else if (m_UseCheckerTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new(new_hittable->mat_ptr->albedo->odd) Texture(Vec3(0.0f, 0.0f, 0.0f), Tex::constant_texture), new(new_hittable->mat_ptr->albedo->even) Texture(Vec3(1.0f, 1.0f, 1.0f), Tex::constant_texture), Tex::checker_texture), m_LightIntensity, Mat::diffuse_light), Hitt::xy_rect));
            }
            else if (m_UseImageTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new_hittable->mat_ptr->albedo->data, m_TextureImageWidth, m_TextureImageHeight, Tex::image_texture), m_LightIntensity, Mat::diffuse_light), Hitt::xy_rect));
            }
        }
        else if (m_UseDielectric) {
            m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(m_IR, Mat::dielectric), Hitt::xy_rect));
        }
    }
    else if (m_UseHittableXZRect) {
        if (m_UseLambertian) {
            if (m_UseConstantTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(m_newColor, Tex::constant_texture), Mat::lambertian), Hitt::xz_rect));
            }
            else if (m_UseCheckerTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new(new_hittable->mat_ptr->albedo->odd) Texture(Vec3(0.0f, 0.0f, 0.0f), Tex::constant_texture), new(new_hittable->mat_ptr->albedo->even) Texture(Vec3(1.0f, 1.0f, 1.0f), Tex::constant_texture), Tex::checker_texture), Mat::lambertian), Hitt::xz_rect));
            }
            else if (m_UseImageTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new_hittable->mat_ptr->albedo->data, m_TextureImageWidth, m_TextureImageHeight, Tex::image_texture), Mat::lambertian), Hitt::xz_rect));
            }
        }
        else if (m_UseMetal) {
            if (m_UseConstantTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(m_newColor, Tex::constant_texture), m_Fuzz, Mat::metal), Hitt::xz_rect));
            }
            else if (m_UseCheckerTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new(new_hittable->mat_ptr->albedo->odd) Texture(Vec3(0.0f, 0.0f, 0.0f), Tex::constant_texture), new(new_hittable->mat_ptr->albedo->even) Texture(Vec3(1.0f, 1.0f, 1.0f), Tex::constant_texture), Tex::checker_texture), m_Fuzz, Mat::metal), Hitt::xz_rect));
            }
            else if (m_UseImageTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new_hittable->mat_ptr->albedo->data, m_TextureImageWidth, m_TextureImageHeight, Tex::image_texture), m_Fuzz, Mat::metal), Hitt::xz_rect));
            }
        }
        else if (m_UseDiffuseLight) {
            if (m_UseConstantTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(m_newColor, Tex::constant_texture), m_LightIntensity, Mat::diffuse_light), Hitt::xz_rect));
            }
            else if (m_UseCheckerTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new(new_hittable->mat_ptr->albedo->odd) Texture(Vec3(0.0f, 0.0f, 0.0f), Tex::constant_texture), new(new_hittable->mat_ptr->albedo->even) Texture(Vec3(1.0f, 1.0f, 1.0f), Tex::constant_texture), Tex::checker_texture), m_LightIntensity, Mat::diffuse_light), Hitt::xz_rect));
            }
            else if (m_UseImageTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new_hittable->mat_ptr->albedo->data, m_TextureImageWidth, m_TextureImageHeight, Tex::image_texture), m_LightIntensity, Mat::diffuse_light), Hitt::xz_rect));
            }
        }
        else if (m_UseDielectric) {
            m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(m_IR, Mat::dielectric), Hitt::xz_rect));
        }
    }
    else if (m_UseHittableYZRect) {
        if (m_UseLambertian) {
            if (m_UseConstantTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(m_newColor, Tex::constant_texture), Mat::lambertian), Hitt::yz_rect));
            }
            else if (m_UseCheckerTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new(new_hittable->mat_ptr->albedo->odd) Texture(Vec3(0.0f, 0.0f, 0.0f), Tex::constant_texture), new(new_hittable->mat_ptr->albedo->even) Texture(Vec3(1.0f, 1.0f, 1.0f), Tex::constant_texture), Tex::checker_texture), Mat::lambertian), Hitt::yz_rect));
            }
            else if (m_UseImageTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new_hittable->mat_ptr->albedo->data, m_TextureImageWidth, m_TextureImageHeight, Tex::image_texture), Mat::lambertian), Hitt::yz_rect));
            }
        }
        else if (m_UseMetal) {
            if (m_UseConstantTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(m_newColor, Tex::constant_texture), m_Fuzz, Mat::metal), Hitt::yz_rect));
            }
            else if (m_UseCheckerTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new(new_hittable->mat_ptr->albedo->odd) Texture(Vec3(0.0f, 0.0f, 0.0f), Tex::constant_texture), new(new_hittable->mat_ptr->albedo->even) Texture(Vec3(1.0f, 1.0f, 1.0f), Tex::constant_texture), Tex::checker_texture), m_Fuzz, Mat::metal), Hitt::yz_rect));
            }
            else if (m_UseImageTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new_hittable->mat_ptr->albedo->data, m_TextureImageWidth, m_TextureImageHeight, Tex::image_texture), m_Fuzz, Mat::metal), Hitt::yz_rect));
            }
        }
        else if (m_UseDiffuseLight) {
            if (m_UseConstantTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(m_newColor, Tex::constant_texture), m_LightIntensity, Mat::diffuse_light), Hitt::yz_rect));
            }
            else if (m_UseCheckerTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new(new_hittable->mat_ptr->albedo->odd) Texture(Vec3(0.0f, 0.0f, 0.0f), Tex::constant_texture), new(new_hittable->mat_ptr->albedo->even) Texture(Vec3(1.0f, 1.0f, 1.0f), Tex::constant_texture), Tex::checker_texture), m_LightIntensity, Mat::diffuse_light), Hitt::yz_rect));
            }
            else if (m_UseImageTexture) {
                m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(new(new_hittable->mat_ptr->albedo) Texture(new_hittable->mat_ptr->albedo->data, m_TextureImageWidth, m_TextureImageHeight, Tex::image_texture), m_LightIntensity, Mat::diffuse_light), Hitt::yz_rect));
            }
        }
        else if (m_UseDielectric) {
            m_World->Add(new(new_hittable) Hittable(m_HittablePosition, m_RectWidth, m_RectHeight, new(new_hittable->mat_ptr) Material(m_IR, Mat::dielectric), Hitt::yz_rect));
        }
    }
}

void CudaLayer::DeleteHittable(Hittable* hittable)
{
    if (hittable->mat_ptr->albedo != nullptr) {
        checkCudaErrors(cudaFree(hittable->mat_ptr->albedo->odd));
        checkCudaErrors(cudaFree(hittable->mat_ptr->albedo->even));
        checkCudaErrors(cudaFree(hittable->mat_ptr->albedo->data));
    }
    checkCudaErrors(cudaFree(hittable->mat_ptr->albedo));
    checkCudaErrors(cudaFree(hittable->mat_ptr));
    checkCudaErrors(cudaFree(hittable));
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