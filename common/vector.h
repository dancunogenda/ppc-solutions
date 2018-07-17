#ifndef VECTOR_H
#define VECTOR_H

#include <cstdlib>

#ifdef __clang__
#define VEC(type,name,n) typedef type name __attribute__ ((ext_vector_type (n)))
#else
#define VEC(type,name,n) typedef type name __attribute__ ((__vector_size__ ((n)*sizeof(type))))
#endif

VEC(float, float4_t, 4);
VEC(float, float8_t, 8);
VEC(double, double4_t, 4);
constexpr float4_t float4_0 = {0,0,0,0};
constexpr float8_t float8_0 = {0,0,0,0,0,0,0,0};
constexpr double4_t double4_0 = {0,0,0,0};

float4_t* float4_alloc(std::size_t n);
float8_t* float8_alloc(std::size_t n);
double4_t* double4_alloc(std::size_t n);

#endif
