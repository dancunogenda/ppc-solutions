#include <cassert>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include "error.h"
#include "timer.h"
#include "so.h"

std::mt19937_64 rng;

enum {
    VAR_RAND,
    VAR_RAND_SMALL,
    VAR_CONST,
    VAR_INCR,
    VAR_DECR,
    N_VAR
};

constexpr data_t MAGIC = 12345;

static void generate(int var, int n, data_t* data) {
    std::uniform_int_distribution<data_t> unif(0, n-1);
    switch (var) {
    case VAR_RAND:
        for (int i = 0; i < n; ++i) {
            data[i] = rng();
        }
        break;
    case VAR_RAND_SMALL:
        for (int i = 0; i < n; ++i) {
            data[i] = rng() & 3;
        }
        break;
    case VAR_CONST:
        for (int i = 0; i < n; ++i) {
            data[i] = MAGIC;
        }
        break;
    case VAR_INCR:
        for (int i = 0; i < n; ++i) {
            data[i] = MAGIC + i;
        }
        break;
    case VAR_DECR:
        for (int i = 0; i < n; ++i) {
            data[i] = MAGIC + n - i;
        }
        break;
    default:
        assert(false);
    }
}

static void copy(int n, const data_t* x, data_t* y) {
    for (int i = 0; i < n; ++i) {
        y[i] = x[i];
    }
}

static void check(int n, const data_t* x, const data_t* y) {
    for (int i = 0; i < n; ++i) {
        if (x[i] != y[i]) {
            std::cerr << "error: element " << i << " differs: "
                << x[i] << " vs. " << y[i] << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, const char** argv) {
    if (argc != 2) {
        error("usage: so-test N");
    }
    int n = std::stoi(argv[1]);
    if (n < 1) {
        error("N has to be positive");
    }
    std::vector<data_t> data1(n);
    std::vector<data_t> data2(n);
    for (int v = 0; v < N_VAR; ++v) {
        generate(v, n, data1.data());
        copy(n, data1.data(), data2.data());
        std::cout << "so\t" << v << "\t" << n << "\t" << std::flush;
        { Timer t(true); psort(n, data2.data()); }
        { Timer t(true); std::sort(data1.data(), data1.data() + n); }
        std::cout << std::endl;
        check(n, data1.data(), data2.data());
    }
}
