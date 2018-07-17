#ifndef VECIM_H
#define VECIM_H

#include <cassert>
#include "image.h"
#include "vector.h"

inline float4_t getvec(const Image8& im, int y, int x) {
    assert(im.nc == 3);
    float4_t v = { im.getlin(y, x, 0), im.getlin(y, x, 1), im.getlin(y, x, 2), 0.0f };
    return v;
}

inline void setvec(Image8& im, int y, int x, float4_t v) {
    assert(im.nc == 3);
    im.setlin(y, x, 0, v[0]);
    im.setlin(y, x, 1, v[1]);
    im.setlin(y, x, 2, v[2]);
}

inline void fillvec(Image8& im, int y0, int x0, int y1, int x1, float4_t v) {
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            setvec(im, y, x, v);
        }
    }
}

#endif
