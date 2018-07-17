#include <cassert>
#include <iostream>
#include "mf.h"

static void compare(const char* file, int line, int l, const float* out, const float* expected) {
    for (int i = 0; i < l; ++i) {
        if (out[i] != expected[i]) {
            std::cerr << file << ":" << line << ": mismatch: " 
                << i << ": " << out[i] << " vs. " << expected[i] << "\n";
            std::exit(EXIT_FAILURE);
        }
    }
}

static void check(const char *file, int line, int ny, int nx, int hy, int hx,
                  const float* in, float* out, const float* expected) {
    for (int i = 0; i < nx * ny; ++i) {
        assert(in[i] >= 0.0);
        assert(in[i] <= 1.0);
        assert(expected[i] >= 0.0);
        assert(expected[i] <= 1.0);
    }
    mf(ny, nx, hy, hx, in, out);
    compare(file, line, nx * ny, out, expected);
}

#define T(ny,nx,hy,hx,in,out,exp) check(__FILE__,__LINE__,ny,nx,hy,hx,in,out,exp)

static void test0() {
    const int ny = 1;
    const int nx = 1;
    const int n = ny * nx;
    float in[n] = {0};
    float out[n];
    T(ny, nx, 0, 0, in, out, in);
    T(ny, nx, 1, 1, in, out, in);
    T(ny, nx, 0, 100, in, out, in);
    T(ny, nx, 100, 0, in, out, in);
    T(ny, nx, 100, 100, in, out, in);
    in[0] = 1;
    T(ny, nx, 0, 0, in, out, in);
    T(ny, nx, 1, 1, in, out, in);
    T(ny, nx, 0, 100, in, out, in);
    T(ny, nx, 100, 0, in, out, in);
    T(ny, nx, 100, 100, in, out, in);
}

static void test1() {
    const int ny = 5;
    const int nx = 10;
    const int n = ny * nx;
    const float A = 0.25;
    const float B = 0.5;
    const float C = 1.0;
    float in[n] = {
        0,0,0,B,B,B,0,0,0,0,
        0,0,0,B,B,B,0,0,0,0,
        0,0,0,B,B,0,0,0,0,0,
        0,0,0,B,B,B,0,0,C,0,
        0,0,0,B,B,B,0,0,0,0
    };
    float zero[n] = {
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0
    };
    float exp01[n] = {
        0,0,0,B,B,B,0,0,0,0,
        0,0,0,B,B,B,0,0,0,0,
        0,0,0,B,B,B,0,0,0,0,
        0,0,0,B,B,B,0,0,0,0,
        0,0,0,B,B,B,0,0,B,0
    };
    float exp10[n] = {
        0,0,0,B,B,B,0,0,0,0,
        0,0,0,B,B,B,0,0,0,0,
        0,0,0,B,B,0,0,0,0,0,
        0,0,0,B,B,B,0,0,0,B,
        0,0,0,B,B,B,0,0,0,0
    };
    float exp20[n] = {
        0,0,0,B,B,B,0,0,0,0,
        0,0,0,B,B,B,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,B,B,B,B,0,0,0,
        0,0,0,B,B,B,0,0,0,0
    };
    float exp30[n] = {
        0,0,A,0,0,0,0,0,0,0,
        0,0,A,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,A,0,0,B,B,A,0,0,
        0,0,A,0,0,0,0,0,0,0
    };
    float exp40[n] = {
        0,A,0,0,0,0,0,0,0,0,
        0,A,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,A,0,0,0,0,A,B,A,0,
        0,A,0,0,0,0,0,0,0,0
    };
    float exp50[n] = {
        A,0,0,0,0,0,0,0,0,0,
        A,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        A,0,0,0,0,0,0,A,B,A,
        A,0,0,0,0,0,0,0,0,0
    };
    float exp11[n] = {
        0,0,0,B,B,B,0,0,0,0,
        0,0,0,B,B,B,0,0,0,0,
        0,0,0,B,B,B,0,0,0,0,
        0,0,0,B,B,B,0,0,0,0,
        0,0,0,B,B,B,0,0,0,0
    };
    float exp1[n] = {
        0,0,0,B,B,B,0,0,0,0,
        0,0,0,B,B,B,0,0,0,0,
        0,0,0,B,B,0,0,0,0,0,
        0,0,0,B,B,B,0,0,0,0,
        0,0,0,B,B,B,0,0,0,0
    };
    float exp2[n] = {
        0,0,0,B,B,B,0,0,0,0,
        0,0,0,B,B,B,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,
        0,0,0,B,B,B,B,0,0,0,
        0,0,0,B,B,B,0,0,0,0
    };
    float out[n];
    T(ny, nx, 0, 0, in, out, in);
    T(ny, nx, 1, 0, in, out, exp01);
    T(ny, nx, 2, 0, in, out, exp11);
    T(ny, nx, 3, 0, in, out, exp11);
    T(ny, nx, 4, 0, in, out, exp11);
    T(ny, nx,99, 0, in, out, exp11);
    T(ny, nx, 0, 1, in, out, exp10);
    T(ny, nx, 0, 2, in, out, exp20);
    T(ny, nx, 0, 3, in, out, exp30);
    T(ny, nx, 0, 4, in, out, exp40);
    T(ny, nx, 0, 5, in, out, exp50);
    T(ny, nx, 0, 8, in, out, zero);
    T(ny, nx, 0,99, in, out, zero);
    T(ny, nx, 1, 1, in, out, exp11);
    T(ny, nx, 2, 2, in, out, exp11);
    T(ny, nx, 4, 4, in, out, zero);
    T( 1,  n, 0, 1, in, out, exp1);
    T( 1,  n, 1, 1, in, out, exp1);
    T( 1,  n,99, 1, in, out, exp1);
    T( 1,  n, 0, 2, in, out, exp2);
    T( 1,  n, 1, 2, in, out, exp2);
    T( 1,  n,99, 2, in, out, exp2);
    T( n,  1, 1, 0, in, out, exp1);
    T( n,  1, 1, 1, in, out, exp1);
    T( n,  1, 1,99, in, out, exp1);
    T( n,  1, 2, 0, in, out, exp2);
    T( n,  1, 2, 1, in, out, exp2);
    T( n,  1, 2,99, in, out, exp2);
}

int main() {
    test0();
    test1();
}
