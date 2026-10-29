#include "stealth/model_file.hpp"
#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#endif

using namespace stealth::numeric;
namespace {
using Clock = std::chrono::steady_clock;
static_assert(Clock::is_steady);
constexpr std::size_t limit = 16 * 1024 * 1024;
constexpr double absolute_tolerance = 5e-5, relative_tolerance = 2e-4;
double elapsed(Clock::time_point begin) {
    return std::chrono::duration<double, std::nano>(Clock::now() - begin).count();
}
long long peak_rss() {
#if defined(__APPLE__) || defined(__linux__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0)
        return -1;
#if defined(__APPLE__)
    return static_cast<long long>(usage.ru_maxrss);
#else
    return static_cast<long long>(usage.ru_maxrss) * 1024;
#endif
#else
    return -1;
#endif
}
void emit_rss(long long value) {
    if (value < 0)
        std::cout << "null";
    else
        std::cout << value;
}
std::vector<unsigned char> read(const std::string &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto size = file.tellg();
    if (!file || size < 0 || size > static_cast<std::streamoff>(limit))
        throw std::runtime_error("Invalid benchmark file size");
    std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char *>(bytes.data()), size))
        throw std::runtime_error("Benchmark file read failed");
    return bytes;
}
std::uint32_t u32(const std::vector<unsigned char> &b, std::size_t i) {
    if (i > b.size() || b.size() - i < 4)
        throw std::runtime_error("Truncated integer");
    return static_cast<std::uint32_t>(b[i]) | (static_cast<std::uint32_t>(b[i + 1]) << 8) |
           (static_cast<std::uint32_t>(b[i + 2]) << 16) |
           (static_cast<std::uint32_t>(b[i + 3]) << 24);
}
std::vector<float> floats(const std::vector<unsigned char> &bytes, std::size_t start,
                          std::size_t n) {
    if (start + n * 4 != bytes.size())
        throw std::runtime_error("Float payload length mismatch");
    std::vector<float> values(n);
    for (std::size_t i = 0; i < n; ++i) {
        values[i] = std::bit_cast<float>(u32(bytes, start + 4 * i));
        if (!std::isfinite(values[i]))
            throw std::runtime_error("Nonfinite float payload");
    }
    return values;
}
std::size_t number(const std::string &s, std::size_t maximum) {
    std::size_t n = 0;
    const auto result = std::from_chars(s.data(), s.data() + s.size(), n);
    if (result.ec != std::errc{} || result.ptr != s.data() + s.size() || n == 0 || n > maximum)
        throw std::runtime_error("Invalid bounded positive count");
    return n;
}
void check(Error error) {
    if (error != Error::ok)
        throw std::runtime_error(error_name(error));
}
} // namespace
int main(int argc, char **argv) {
    try {
        const auto baseline_rss = peak_rss();
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc || !args.emplace(argv[i], argv[i + 1]).second)
                throw std::runtime_error("Expected unique option/value pairs");
        }
        if (args.size() != 6 || !args.contains("--model") || !args.contains("--input") ||
            !args.contains("--expected") || !args.contains("--batch") ||
            !args.contains("--samples") || !args.contains("--warmup"))
            throw std::runtime_error(
                "Required: --model --input --expected --batch --samples --warmup");
        const auto batch = number(args.at("--batch"), 1024),
                   samples = number(args.at("--samples"), 1000000),
                   warmup = number(args.at("--warmup"), 100000);
        const auto setup_begin = Clock::now();
        const auto model = load_forecaster_file(args.at("--model"));
        const auto c = model->config();
        ForecastWorkspace workspace(c);
        const auto bytes = read(args.at("--input"));
        const std::string magic("STLBN01\0", 8);
        if (bytes.size() < 20 || !std::equal(magic.begin(), magic.end(), bytes.begin()))
            throw std::runtime_error("Invalid input header");
        const std::size_t windows = u32(bytes, 8), tokens = u32(bytes, 12),
                          features = u32(bytes, 16);
        if (windows < 1 || windows > 1024 || tokens < 1 || tokens > c.context ||
            features != c.features || windows % batch != 0 ||
            windows * tokens * features > (limit - 20) / 4)
            throw std::runtime_error("Invalid benchmark dimensions");
        const auto input = floats(bytes, 20, windows * tokens * features);
        const auto expected = floats(read(args.at("--expected")), 0, windows * tokens);
        std::vector<float> output(batch * tokens);
        const auto setup_ns = elapsed(setup_begin);
        const auto setup_rss = peak_rss();
        const std::span<const float> inputs(input);
        const std::span<float> outputs(output);
        const auto groups = windows / batch;
        const auto forward = [&](std::size_t group) {
            for (std::size_t j = 0; j < batch; ++j)
                check(model->forward(
                    inputs.subspan((group * batch + j) * tokens * features, tokens * features),
                    tokens, outputs.subspan(j * tokens, tokens), workspace));
        };
        const auto first_begin = Clock::now();
        forward(0);
        const double first_call_ns = elapsed(first_begin);
        double max_difference = 0;
        // Every origin of every fixed window passes before any warm timings.
        for (std::size_t group = 0; group < groups; ++group) {
            forward(group);
            for (std::size_t j = 0; j < output.size(); ++j) {
                const double e = expected[group * batch * tokens + j], a = output[j];
                const double difference = std::abs(a - e);
                if (!std::isfinite(a) ||
                    difference > absolute_tolerance + relative_tolerance * std::abs(e))
                    throw std::runtime_error("Runtime parity failed");
                max_difference = std::max(max_difference, difference);
            }
        }
        for (std::size_t i = 0; i < warmup; ++i)
            forward(i % groups);
        std::vector<double> times(samples), clock_times(1000);
        double checksum = 0;
        for (auto &t : clock_times) {
            const auto begin = Clock::now();
            t = elapsed(begin);
        }
        for (std::size_t i = 0; i < samples; ++i) {
            const auto begin = Clock::now();
            forward(i % groups);
            times[i] = elapsed(begin);
            checksum += output[i % output.size()];
        }
        const auto final_rss = peak_rss();
        if (!std::isfinite(checksum))
            throw std::runtime_error("Nonfinite checksum");
        std::cout << std::setprecision(17) << "{\"engine\":\"native_core\",\"batch\":" << batch
                  << ",\"setup_ns\":" << setup_ns << ",\"first_call_ns\":" << first_call_ns
                  << ",\"max_abs_difference\":" << max_difference << ",\"checksum\":" << checksum
                  << ",\"parameter_bytes\":" << model->parameter_bytes()
                  << ",\"position_bytes\":" << model->position_bytes()
                  << ",\"workspace_bytes\":" << workspace.bytes()
                  << ",\"peak_rss_bytes\":{\"baseline\":";
        emit_rss(baseline_rss);
        std::cout << ",\"setup\":";
        emit_rss(setup_rss);
        std::cout << ",\"final\":";
        emit_rss(final_rss);
        std::cout << "},\"times_ns\":[";
        for (std::size_t i = 0; i < times.size(); ++i) {
            if (i)
                std::cout << ',';
            std::cout << times[i];
        }
        std::cout << "],\"clock_times_ns\":[";
        for (std::size_t i = 0; i < clock_times.size(); ++i) {
            if (i)
                std::cout << ',';
            std::cout << clock_times[i];
        }
        std::cout << "]}\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
