#include <iostream>
#include <vector>
#include <random>
#include "error.h"
#include "timer.h"
#include "mf.h"

static void benchmark(int ny, int nx, int k) {
    std::mt19937 rng;
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<float> data(ny * nx);
    std::vector<float> result(ny * nx);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            float v = u(rng);
            data[x + nx * y] = v;
        }
    }
    std::cout << "mf\t" << ny << "\t" << nx << "\t" << k << "\t" << std::flush;
    { Timer t; mf(ny, nx, k, k, data.data(), result.data()); }
    std::cout << std::endl;
}

int main(int argc, const char** argv) {
    if (argc != 4) {
        error("usage: mf-benchmark Y X K");
    }
    int ny = std::stoi(argv[1]);
    int nx = std::stoi(argv[2]);
    int k = std::stoi(argv[3]);
    benchmark(ny, nx, k);
}
