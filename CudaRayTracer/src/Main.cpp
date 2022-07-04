#include "Application.h"

#include "Log.h"

int main(int argc, char** argv)
{
    int a = 5;
    Log::Init();
    RT_INFO("Log Initialized!");

    Application* app = new Application();
    app->Run();
    delete app;

    return 0;
}