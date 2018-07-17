#include <iostream>
#include <vector>
#include <random>
#include "error.h"
#include "timer.h"
#include "cp.h"

static void generate(int ny, int nx, float* data) {
    std::mt19937 rng;
    std::uniform_real_distribution<double> unif(0.0f, 1.0f);
    std::bernoulli_distribution coin(0.2);
    for (int y = 0; y < ny; ++y) {
        if (y > 0 && coin(rng)) {
            // Introduce some correlations
            int row = std::min(static_cast<int>(y * unif(rng)), y - 1);
            double offset = 2.0 * (unif(rng) - 0.5f);
            double mult = 2.0 * unif(rng);
            for (int x = 0; x < nx; ++x) {
                double error = 0.1 * unif(rng);
                data[x + nx * y] = mult * data[x + nx * row] + offset + error;
            }
        } else {
            // Generate random data
            for (int x = 0; x < nx; ++x) {
                double v = unif(rng);
                data[x + nx * y] = v;
            }
        }
    }
}

static void verify(double& worst, int& bad, double maxerr, int ny, int nx, const float* data, const float* result) {
    worst = 0.0;
    bad = 0;
    for (int j = 0; j < ny; ++j) {
        for (int i = j; i < ny; ++i) {
            // naive algorithm
            double q = result[i + ny * j];
            double sa = 0.0;
            double sb = 0.0;
            double sab = 0.0;
            double saa = 0.0;
            double sbb = 0.0;
            for (int x = 0; x < nx; ++x) {
                double a = data[x + nx * i];
                double b = data[x + nx * j];
                sa += a;
                sb += b;
                sab += a * b;
                saa += a * a;
                sbb += b * b;
            }
            double r = nx * sab - sa * sb;
            r /= std::sqrt(nx * saa - sa * sa);
            r /= std::sqrt(nx * sbb - sb * sb);
            double err = std::abs(q - r);
            worst = std::max(err, worst);
            if (err > maxerr) {
                ++bad;
            }
        }
    }
}

static void report(double worst, int bad, double maxerr) {
    // Error margin
    std::cout << "\t" << std::fixed << std::setprecision(4) << worst / maxerr << std::endl;
    std::cout.copyfmt(std::ios(NULL));
    if (bad) {
        std::cerr << "Test failed:\n";
        std::cerr << bad << " pairs with error larger than " << maxerr << "\n";
        std::cerr << "Worst error " << worst << std::endl;
        exit(EXIT_FAILURE);
    }
}

static void test(double maxerr, int ny, int nx) {
    std::vector<float> data(ny * nx);
    generate(ny, nx, data.data());
    std::vector<float> result(ny * ny);
    std::cout << "cp\t" << ny << "\t" << nx << "\t" << std::flush;
    { Timer t; correlate(ny, nx, data.data(), result.data()); }
    double worst = 0.0;
    int bad = 0;
    verify(worst, bad, maxerr, ny, nx, data.data(), result.data());
    report(worst, bad, maxerr);
}

int main(int argc, const char** argv) {
    if (argc != 4) {
        error("usage: cp-test MAXERROR Y X");
    }
    double maxerr = std::stod(argv[1]);
    int ny = std::stoi(argv[2]);
    int nx = std::stoi(argv[3]);
    test(maxerr, ny, nx);
}
