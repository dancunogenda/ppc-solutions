#ifndef TIMER_H
#define TIMER_H

#include <iomanip>
#include <iostream>
#include <sys/time.h>

class Timer {
public:
    Timer(bool add_tab_=false) : start{get_time()}, add_tab{add_tab_}
    {}

    ~Timer() {
        double now = get_time();
        std::cout << std::fixed << std::setprecision(3) << (now - start);
        if (add_tab) {
            std::cout << "\t";
        }
        std::cout << std::flush;
        std::cout.copyfmt(std::ios(NULL));
    }

private:
    static double get_time() {
        struct timeval tm;
        gettimeofday(&tm, NULL);
        return static_cast<double>(tm.tv_sec) + static_cast<double>(tm.tv_usec) / 1E6;
    }

    double start;
    bool add_tab;
};

#endif
