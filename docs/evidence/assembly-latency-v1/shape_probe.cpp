#include "stealth/numeric.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace stealth::numeric;
int main() {
    try {
        double checksum = 0;
        std::cout << "{\"samples\":200,\"calls_per_sample\":16,\"warmup\":64,\"cells\":[";
        bool first = true;
        for (std::size_t inputs : {1U, 3U, 7U, 16U, 64U, 256U})
            for (std::size_t outputs : {1U, 7U, 8U, 9U, 64U, 192U})
                for (std::size_t rows : {1U, 4U, 5U}) {
                    std::vector<float> w(inputs * outputs), b(outputs), x(inputs * rows),
                        y(outputs * rows), ref(outputs * rows), scratch(outputs * rows);
                    for (std::size_t i = 0; i < w.size(); ++i) w[i] = static_cast<float>(static_cast<int>(i % 27) - 13) / 31.1F;
                    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 13.3F;
                    Dense model(inputs, outputs, w, b), scalar(inputs, outputs, w, b, DenseKernel::reference);
                    if (scalar.forward_rows(x, rows, ref, scratch) != Error::ok) throw std::runtime_error("reference");
                    for (int k = 0; k < 64; ++k)
                        if (model.forward_rows(x, rows, y, scratch) != Error::ok) throw std::runtime_error("warmup");
                    for (std::size_t i = 0; i < y.size(); ++i)
                        if (std::bit_cast<std::uint32_t>(y[i]) != std::bit_cast<std::uint32_t>(ref[i])) throw std::runtime_error("parity");
                    std::vector<double> times(200);
                    for (auto &ns : times) {
                        auto begin = std::chrono::steady_clock::now();
                        for (int k = 0; k < 16; ++k)
                            if (model.forward_rows(x, rows, y, scratch) != Error::ok) throw std::runtime_error("forward");
                        ns = std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-begin).count()/16;
                        checksum += y.back();
                    }
                    if (!first) std::cout << ',';
                    first = false;
                    std::cout << "{\"inputs\":" << inputs << ",\"outputs\":" << outputs << ",\"rows\":" << rows << ",\"times_ns\":[";
                    for (std::size_t i = 0; i < times.size(); ++i) { if (i) std::cout << ','; std::cout << times[i]; }
                    std::cout << "]}";
                }
        std::cout << "],\"checksum\":" << checksum << "}\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
