#include "stealth/model_file.hpp"
#include "sha256.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace stealth::numeric {
namespace {
std::uint32_t read32(std::span<const std::byte> bytes, std::size_t offset) noexcept {
    std::uint32_t value = 0;
    for (unsigned int i = 0; i < 4; ++i)
        value |= std::to_integer<std::uint32_t>(bytes[offset + i]) << (8 * i);
    return value;
}
float read_float(std::span<const std::byte> bytes, std::size_t offset) noexcept {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    return std::bit_cast<float>(read32(bytes, offset));
}
} // namespace
std::unique_ptr<ForecastModel> load_forecaster(std::span<const std::byte> bytes,
                                               DenseKernel kernel) {
    if (bytes.size() > max_model_file_bytes)
        throw std::length_error("Model file exceeds byte limit");
    if (bytes.size() < 96)
        throw std::invalid_argument("Truncated model file");
    // Verify all bounded bytes before interpreting header dimensions or payload values.
    const auto digest = stealth::detail::sha256(bytes.first(bytes.size() - 32));
    if (!std::equal(digest.begin(), digest.end(), bytes.end() - 32))
        throw std::invalid_argument("Model integrity digest mismatch");
    constexpr std::array<unsigned char, 8> magic{'S', 'T', 'L', 'F', 'C', '0', '1', 0};
    for (std::size_t i = 0; i < magic.size(); ++i)
        if (std::to_integer<unsigned char>(bytes[i]) != magic[i])
            throw std::invalid_argument("Model magic mismatch");
    if (read32(bytes, 8) != 1 || read32(bytes, 12) != 0x01020304U || read32(bytes, 16) != 64 ||
        read32(bytes, 56) != 0 || read32(bytes, 60) != 0)
        throw std::invalid_argument("Unsupported model version, byte order or fields");
    const ForecastConfig config{read32(bytes, 20), read32(bytes, 24),    read32(bytes, 28),
                                read32(bytes, 32), read32(bytes, 36),    read32(bytes, 40),
                                read32(bytes, 44), read_float(bytes, 48)};
    const auto count = forecast_parameter_count(config); // validates before dimension products
    if (read32(bytes, 52) != count || bytes.size() != 96 + count * 4)
        throw std::invalid_argument("Model payload size/count mismatch");
    std::vector<float> parameters(count);
    for (std::size_t i = 0; i < count; ++i)
        parameters[i] = read_float(bytes, 64 + 4 * i);
    // Constructor checks every value and scale before owning/packing parameters.
    return std::make_unique<ForecastModel>(config, parameters, kernel);
}
std::unique_ptr<ForecastModel> load_forecaster_file(const std::filesystem::path &path,
                                                    DenseKernel kernel) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        throw std::runtime_error("Cannot open model file");
    const auto length = stream.tellg();
    if (length < 0)
        throw std::runtime_error("Cannot measure model file");
    if (length > static_cast<std::streamoff>(max_model_file_bytes))
        throw std::length_error("Model file exceeds byte limit");
    if (length < 96)
        throw std::invalid_argument("Truncated model file");
    std::vector<std::byte> bytes(static_cast<std::size_t>(length));
    stream.seekg(0, std::ios::beg);
    stream.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream || stream.gcount() != static_cast<std::streamsize>(bytes.size()))
        throw std::runtime_error("Short model file read");
    if (stream.peek() != std::char_traits<char>::eof())
        throw std::invalid_argument("Model file grew during read");
    if (stream.bad())
        throw std::runtime_error("Model file read failed");
    return load_forecaster(bytes, kernel);
}
} // namespace stealth::numeric
