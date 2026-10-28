#pragma once
#include <array>
#include <cstddef>
#include <span>
namespace stealth::detail {
// Internal artifact-integrity helper; not an authentication API.
[[nodiscard]] std::array<std::byte, 32> sha256(std::span<const std::byte> input) noexcept;
} // namespace stealth::detail
