#pragma once

#ifdef RT_DEBUG
    #define RT_ENABLE_ASSERTS
#endif

#ifdef RT_ENABLE_ASSERTS
    #if _MSC_VER
        #include <intrin.h>
        #define debugbreak() __debugbreak();
    #else
        #define debugbreak() __builtin_trap();
    #endif // _MSC_VER
    #define RT_ASSERT(x, ...) { if (!(x)) { RT_ERROR("Assertion Failed: {0}", __VA_ARGS__); debugbreak(); }}
#else
    #define RT_ASSERT(x, ...)
#endif // HZ_ENABLE_ASSERTS

#define BIT(x) (1 << x)

#define RT_BIND_EVENT_FN(fn) std::bind(&fn, this, std::placeholders::_1)