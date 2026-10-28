#pragma once
#include "stealth/numeric.hpp"

namespace stealth::numeric {
inline constexpr std::size_t max_context = 1024;
inline constexpr std::size_t max_width = 512;
inline constexpr std::size_t max_hidden = 2048;

class AttentionWorkspace {
  public:
    AttentionWorkspace(std::size_t capacity, std::size_t width, std::size_t heads);
    [[nodiscard]] std::size_t bytes() const noexcept { return storage_.size() * sizeof(float); }

  private:
    friend Error causal_attention(std::span<const float>, std::span<const float>,
                                  std::span<const float>, std::size_t, std::span<float>,
                                  AttentionWorkspace &) noexcept;
    std::size_t capacity_, width_, heads_;
    AlignedBuffer storage_;
};

// Q, K, V and output are row-major [tokens,width]. Contiguous channel groups are heads.
// Each row includes itself and earlier rows. The whole output is unchanged on error.
[[nodiscard]] Error causal_attention(std::span<const float> query, std::span<const float> key,
                                     std::span<const float> value, std::size_t tokens,
                                     std::span<float> output,
                                     AttentionWorkspace &workspace) noexcept;
} // namespace stealth::numeric
