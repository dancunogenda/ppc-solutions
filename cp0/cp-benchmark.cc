#include <iostream>
#include <vector>
#include <random>
#include "error.h"
#include "timer.h"
#include "cp.h"

static void benchmark(int ny, int nx) {
    std::mt19937 rng;
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<float> data(ny * nx);
    std::vector<float> result(ny * ny);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            float v = u(rng);
            data[x + nx * y] = v;
        }
    }
    std::cout << "cp\t" << ny << "\t" << nx << "\t" << std::flush;
    { Timer t; correlate(ny, nx, data.data(), result.data()); }
    std::cout << std::endl;
}

int main(int argc, const char** argv) {
    if (argc != 3) {
        error("usage: cp-benchmark Y X");
    }
    int ny = std::stoi(argv[1]);
    int nx = std::stoi(argv[2]);
    benchmark(ny, nx);
}
