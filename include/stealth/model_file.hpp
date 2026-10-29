#pragma once
#include "stealth/forecaster.hpp"
#include <filesystem>
#include <memory>

namespace stealth::numeric {
inline constexpr std::size_t max_model_file_bytes = 16 * 1024 * 1024;
// Cold operations. Invalid artifacts throw std::invalid_argument/length_error;
// file errors throw std::runtime_error. Allocation failures propagate.
// Publish with active = load_forecaster(...); failed loading preserves active.
[[nodiscard]] std::unique_ptr<ForecastModel>
load_forecaster(std::span<const std::byte> bytes, DenseKernel kernel = DenseKernel::packed);
[[nodiscard]] std::unique_ptr<ForecastModel>
load_forecaster_file(const std::filesystem::path &path, DenseKernel kernel = DenseKernel::packed);
} // namespace stealth::numeric
