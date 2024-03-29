#include "Core/Application.h"

#include "Core/Log.h"

int main(int argc, char** argv)
{
    Log::Init();
    RT_INFO("Log Initialized!");

    Application* app = new Application();
    app->Run();
    delete app;

    return 0;
}
