#pragma once

#include <memory>
#include <spdlog/spdlog.h>

class Log
{
public:
    static void Init();

    inline static std::shared_ptr<spdlog::logger>& GetLogger() { return s_Logger; }

private:
    static std::shared_ptr<spdlog::logger> s_Logger;
};

#if defined(RT_DEBUG) || defined(RT_RELEASE)
#   define RT_TRACE(...)   ::Log::GetLogger()->trace(__VA_ARGS__);
#   define RT_INFO(...)    ::Log::GetLogger()->info(__VA_ARGS__);
#   define RT_WARN(...)    ::Log::GetLogger()->warn(__VA_ARGS__);
#   define RT_ERROR(...)   ::Log::GetLogger()->error(__VA_ARGS__);
#   define RT_FATAL(...)   ::Log::GetLogger()->critical(__VA_ARGS__);
#elif defined(RT_DIST)
#   define RT_TRACE
#   define RT_INFO
#   define RT_WARN
#   define RT_ERROR
#   define RT_FATAL
#endif // RT_DEBUG || RT_RELEASE
