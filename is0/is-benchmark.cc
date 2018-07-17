#include <iostream>
#include <vector>
#include <random>
#include "error.h"
#include "timer.h"
#include "is.h"

static void benchmark(int ny, int nx) {
    std::mt19937 rng;
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<float> data(ny * nx * 3);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            for (int c = 0; c < 3; ++c) {
                float v = u(rng);
                data[c + 3 * x + 3 * nx * y] = v;
            }
        }
    }
    std::cout << "is\t" << ny << "\t" << nx << "\t" << std::flush;
    { Timer t; segment(ny, nx, data.data()); }
    std::cout << std::endl;
}

int main(int argc, const char** argv) {
    if (argc != 3) {
        error("usage: is-benchmark Y X");
    }
    int ny = std::stoi(argv[1]);
    int nx = std::stoi(argv[2]);
    benchmark(ny, nx);
}
