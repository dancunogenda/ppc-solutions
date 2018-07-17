#include "vector.h"
#include <algorithm>

static void* myalloc(std::size_t ts, std::size_t n) {
    void* tmp = 0;
    std::size_t align = std::max(ts, sizeof(void*));
    std::size_t size = ts;
    size *= static_cast<std::size_t>(n);
    if (posix_memalign(&tmp, align, size)) {
        throw std::bad_alloc();
    }
    return tmp;
}


float4_t* float4_alloc(std::size_t n) {
    return static_cast<float4_t*>(myalloc(sizeof(float4_t), n));
}

float8_t* float8_alloc(std::size_t n) {
    return static_cast<float8_t*>(myalloc(sizeof(float8_t), n));
}

double4_t* double4_alloc(std::size_t n) {
    return static_cast<double4_t*>(myalloc(sizeof(double4_t), n));
}
