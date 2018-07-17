#include <cmath>
#include <random>
#include <iostream>
#include <vector>
#include "error.h"
#include "timer.h"
#include "is.h"

typedef std::mt19937 Rng;

static void colours(Rng& rng, float a[3], float b[3]) {
    std::uniform_real_distribution<double> unif(0.0f, 1.0f);
    std::bernoulli_distribution coin(0.2);
    bool done = false;
    while (!done) {
        for (int k = 0; k < 3; ++k) {
            a[k] = unif(rng);
            b[k] = coin(rng) ? unif(rng) : a[k];
            if (a[k] != b[k]) {
                done = true;
            }
        }
    }
}

static void dump(const float a[3]) {
    std::cout << a[0] << "," << a[1] << "," << a[2];
}

static void dump(const Result& r) {
    std::cout << "  y0 = " << r.y0 << "\n";
    std::cout << "  x0 = " << r.x0 << "\n";
    std::cout << "  y1 = " << r.y1 << "\n";
    std::cout << "  x1 = " << r.x1 << "\n";
    std::cout << "  outer = "; dump(r.outer); std::cout << "\n";
    std::cout << "  inner = "; dump(r.inner); std::cout << "\n";
}

static bool close(float a, float b) {
    return std::abs(a - b) < 0.0001;
}

static bool equal(const float a[3], const float b[3]) {
    return close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2]);
}

static void compare(int ny, int nx, const Result& e, const Result& r) {
    if (e.y0 == r.y0 && e.x0 == r.x0 && e.y1 == r.y1 && e.x1 == r.x1 &&
        equal(e.outer, r.outer) && equal(e.inner, r.inner)) {
        return;
    }
    std::cerr << "Test failed." << std::endl;
    std::cout << "ny = " << ny << "\n";
    std::cout << "nx = " << nx << "\n";
    std::cout << "Expected:\n";
    dump(e);
    std::cout << "Got:\n";
    dump(r);
    exit(EXIT_FAILURE);
}

static void test(Rng& rng, int ny, int nx, int y0, int x0, int y1, int x1) {
    Result e;
    e.y0 = y0;
    e.x0 = x0;
    e.y1 = y1;
    e.x1 = x1;
    // Random but distinct colours
    colours(rng, e.inner, e.outer);
    std::vector<float> data(3*ny*nx);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            for (int c = 0; c < 3; ++c) {
                bool inside = y0 <= y && y < y1 && x0 <= x && x < x1;
                data[c + 3 * x + 3 * nx * y] = inside ? e.inner[c] : e.outer[c];
            }
        }
    }
    Result r;
    std::cout << "is\t" << ny << "\t" << nx << "\t" << std::flush;
    { Timer t; r = segment(ny, nx, data.data()); }
    std::cout << std::endl;
    compare(ny, nx, e, r);
}

static void test(Rng& rng, int ny, int nx) {
    for (int i = 0; i < 10; ++i) {
        // Random box location
        std::uniform_real_distribution<> dy0(0, ny-1);  int y0 = dy0(rng);
        std::uniform_real_distribution<> dx0(0, nx-1);  int x0 = dx0(rng);
        std::uniform_real_distribution<> dy1(y0+1, ny); int y1 = dy1(rng);
        std::uniform_real_distribution<> dx1(x0+1, nx); int x1 = dx1(rng);
        // Avoid ambiguous cases
        if (y0 == 0 && y1 == ny && x0 == 0)  { continue; }
        if (y0 == 0 && y1 == ny && x1 == nx) { continue; }
        if (x0 == 0 && x1 == nx && y0 == 0)  { continue; }
        if (x0 == 0 && x1 == nx && y1 == ny) { continue; }
        test(rng, ny, nx, y0, x0, y1, x1);
    }
}

int main() {
    Rng rng;
    for (int ny = 1; ny < 60; ++ny) {
        test(rng, ny, 1);
        test(rng, 1, ny);
    }
    for (int ny = 2; ny < 60; ny += 13) {
        for (int nx = 2; nx < 60; nx += 7) {
            test(rng, ny, nx);
        }
    }
    test(rng, 1000, 1);
    test(rng, 1, 1000);
    test(rng, 1000, 2);
    test(rng, 2, 1000);
    test(rng, 100, 50);
}
