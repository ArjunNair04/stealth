#include "stealth/feed_forward.hpp"
#include "stealth/runtime.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace stealth::numeric;
namespace {
using Clock = std::chrono::steady_clock;
static_assert(Clock::is_steady);
struct Statistics {
    double p50, p95, p99, mean;
};
Statistics summarize(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    double total = 0;
    for (double value : values)
        total += value;
    const auto quantile = [&](double q) {
        const auto index =
            static_cast<std::size_t>(std::ceil(q * static_cast<double>(values.size()))) - 1;
        return values[index];
    };
    return {quantile(0.50), quantile(0.95), quantile(0.99),
            total / static_cast<double>(values.size())};
}
void emit(const Statistics &stats) {
    std::cout << "{\"p50_ns\":" << stats.p50 << ",\"p95_ns\":" << stats.p95
              << ",\"p99_ns\":" << stats.p99 << ",\"mean_ns\":" << stats.mean << '}';
}
template <typename Function> double timed(Function &&function) {
    const auto begin = Clock::now();
    const auto status = function();
    const auto end = Clock::now();
    if (status != Error::ok)
        throw std::runtime_error(error_name(status));
    return std::chrono::duration<double, std::nano>(end - begin).count();
}
template <typename Reference, typename Packed>
void pair(std::size_t samples, Reference &&reference, Packed &&packed,
          std::span<const float> reference_output, std::span<const float> packed_output) {
    for (int i = 0; i < 1000; ++i) {
        if (reference() != Error::ok || packed() != Error::ok)
            throw std::runtime_error("Warm-up failed");
    }
    double max_difference = 0;
    for (std::size_t i = 0; i < reference_output.size(); ++i) {
        const double difference =
            std::abs(static_cast<double>(reference_output[i]) - packed_output[i]);
        max_difference = std::max(max_difference, difference);
        if (!std::isfinite(reference_output[i]) || !std::isfinite(packed_output[i]) ||
            difference > 2e-5 + 5e-5 * std::abs(reference_output[i]))
            throw std::runtime_error("Kernel parity failed");
    }
    std::vector<double> ref_times(samples), packed_times(samples);
    double checksum = 0;
    for (std::size_t i = 0; i < samples; ++i) {
        // Alternate order to reduce systematic drift; this is a warmed microbenchmark.
        if (i % 2 == 0) {
            ref_times[i] = timed(reference);
            packed_times[i] = timed(packed);
        } else {
            packed_times[i] = timed(packed);
            ref_times[i] = timed(reference);
        }
        checksum +=
            reference_output[i % reference_output.size()] + packed_output[i % packed_output.size()];
    }
    const auto ref = summarize(ref_times);
    const auto pack = summarize(packed_times);
    std::cout << ",\"reference\":";
    emit(ref);
    std::cout << ",\"packed\":";
    emit(pack);
    std::cout << ",\"median_speedup\":";
    if (pack.p50 > 0)
        std::cout << ref.p50 / pack.p50;
    else
        std::cout << "null";
    std::cout << ",\"max_abs_difference\":" << max_difference << ",\"checksum\":" << checksum
              << '}';
}
void fill(std::span<float> values, std::size_t seed) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(static_cast<int>((i * 37 + seed) % 257) - 128) / 512.0F;
    }
}
void dense(std::size_t inputs, std::size_t outputs, std::size_t samples) {
    AlignedBuffer w(inputs * outputs), b(outputs), x(inputs), a(outputs), z(outputs), sa(outputs),
        sz(outputs);
    fill(w.span(), 13);
    fill(b.span(), 7);
    fill(x.span(), 3);
    Dense reference(inputs, outputs, w.span(), b.span(), DenseKernel::reference);
    Dense packed(inputs, outputs, w.span(), b.span(), DenseKernel::packed);
    std::cout << "{\"kind\":\"dense\",\"inputs\":" << inputs << ",\"outputs\":" << outputs
              << ",\"reference_parameter_bytes\":" << reference.parameter_bytes()
              << ",\"packed_parameter_bytes\":" << packed.parameter_bytes();
    pair(
        samples, [&] { return reference.forward(x.span(), a.span(), sa.span()); },
        [&] { return packed.forward(x.span(), z.span(), sz.span()); }, a.span(), z.span());
}
void feed_forward(std::size_t width, std::size_t hidden, std::size_t samples) {
    AlignedBuffer scale(width), shift(width), w1(width * hidden), b1(hidden), w2(width * hidden),
        b2(width), x(width), a(width), b(width);
    std::fill(scale.span().begin(), scale.span().end(), 1.0F);
    fill(w1.span(), 7);
    fill(b1.span(), 13);
    fill(w2.span(), 19);
    fill(b2.span(), 3);
    fill(x.span(), 11);
    FeedForwardParameters p{scale.span(), shift.span(), w1.span(), b1.span(), w2.span(), b2.span()};
    FeedForward reference(width, hidden, p, 1e-5F, DenseKernel::reference),
        packed(width, hidden, p);
    FeedForwardWorkspace ra(width, hidden), pa(width, hidden);
    std::cout << "{\"kind\":\"feed_forward\",\"width\":" << width << ",\"hidden\":" << hidden
              << ",\"reference_parameter_bytes\":" << reference.parameter_bytes()
              << ",\"packed_parameter_bytes\":" << packed.parameter_bytes()
              << ",\"workspace_bytes\":" << pa.bytes();
    pair(
        samples, [&] { return reference.forward(x.span(), a.span(), ra); },
        [&] { return packed.forward(x.span(), b.span(), pa); }, a.span(), b.span());
}
template <typename Function>
void component(std::string_view name, std::size_t size, std::size_t samples, Function &&function,
               std::span<const float> output) {
    for (int i = 0; i < 1000; ++i) {
        if (function() != Error::ok)
            throw std::runtime_error("Component warm-up failed");
    }
    std::vector<double> durations(samples);
    double checksum = 0;
    for (std::size_t i = 0; i < samples; ++i) {
        durations[i] = timed(function);
        checksum += output[i % output.size()];
    }
    std::cout << "{\"kind\":\"" << name << "\",\"elements\":" << size << ",\"latency\":";
    emit(summarize(durations));
    std::cout << ",\"checksum\":" << checksum << '}';
}
void components(std::size_t samples) {
    AlignedBuffer x(256), scale(256), shift(256), out(256), scratch(256);
    fill(x.span(), 11);
    std::fill(scale.span().begin(), scale.span().end(), 1.0F);
    component(
        "layer_norm", 64, samples,
        [&] {
            return layer_norm(x.span().first(64), scale.span().first(64), shift.span().first(64),
                              1e-5F, out.span().first(64), scratch.span());
        },
        out.span().first(64));
    std::cout << ',';
    component(
        "gelu", 256, samples, [&] { return gelu(x.span(), out.span(), scratch.span()); },
        out.span());
    std::cout << ',';
    component(
        "softmax", 128, samples,
        [&] { return softmax(x.span().first(128), 128, out.span().first(128), scratch.span()); },
        out.span().first(128));
    std::cout << ',';
    component(
        "residual", 64, samples,
        [&] {
            return residual_add(x.span().first(64), shift.span().first(64), out.span().first(64),
                                scratch.span());
        },
        out.span().first(64));
}
} // namespace
int main(int argc, char *argv[]) {
    try {
        std::size_t samples = 10000;
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout << "Usage: stealth_bench [--samples N] (100..1000000)\n";
            return 0;
        }
        if (argc != 1) {
            if (argc != 3 || std::string_view(argv[1]) != "--samples")
                throw std::runtime_error("Expected --samples N");
            const std::string_view value(argv[2]);
            const auto [end, error] =
                std::from_chars(value.data(), value.data() + value.size(), samples);
            if (error != std::errc{} || end != value.data() + value.size())
                throw std::runtime_error("Invalid sample count");
        }
        if (samples < 100 || samples > 1000000)
            throw std::runtime_error("Sample count outside bounds");
        const auto info = stealth::build_info();
        if (info.configuration != "Release")
            throw std::runtime_error("Benchmark requires a Release build");
        std::vector<double> clock_times(samples);
        for (auto &duration : clock_times) {
            const auto first = Clock::now();
            const auto second = Clock::now();
            duration = std::chrono::duration<double, std::nano>(second - first).count();
        }
        std::cout << std::setprecision(10) << "{\"schema_version\":1,\"compiler\":\""
                  << info.compiler << "\",\"architecture\":\"" << info.architecture
                  << "\",\"configuration\":\"" << info.configuration
                  << "\",\"threads\":1,\"samples_per_kernel\":" << samples
                  << ",\"warmup_calls_per_kernel\":1000,\"measurement\":\"warm per-call wall "
                     "latency including validation; alternating kernel order\",\"clock_overhead\":";
        emit(summarize(clock_times));
        std::cout << ",\"cases\":[";
        dense(3, 4, samples);
        std::cout << ',';
        dense(64, 64, samples);
        std::cout << ',';
        dense(64, 256, samples);
        std::cout << ',';
        dense(256, 64, samples);
        std::cout << ',';
        feed_forward(64, 256, samples);
        std::cout << "],\"components\":[";
        components(samples);
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
