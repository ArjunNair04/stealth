#pragma once
#include "stealth/transformer_block.hpp"

namespace stealth::numeric {
inline constexpr std::size_t max_features = 64;
enum class ForecastOutput { level, correction };
struct ForecastConfig {
    std::size_t features, width, heads, hidden, context, horizon, target;
    float epsilon = 1e-5F;
    ForecastOutput output = ForecastOutput::level;
    bool operator==(const ForecastConfig &) const = default;
};
// Validates config before computing any dimension product.
[[nodiscard]] std::size_t forecast_parameter_count(ForecastConfig config);
class ForecastWorkspace {
  public:
    explicit ForecastWorkspace(ForecastConfig config);
    [[nodiscard]] std::size_t bytes() const noexcept;

  private:
    friend class ForecastModel;
    ForecastConfig config_;
    AlignedBuffer storage_;
    TransformerWorkspace block_;
};
class ForecastModel {
  public:
    // Canonical flat order is specified in docs/forecast-contract.md.
    ForecastModel(ForecastConfig config, std::span<const float> parameters,
                  DenseKernel kernel = DenseKernel::packed);
    [[nodiscard]] ForecastConfig config() const noexcept { return config_; }
    [[nodiscard]] std::size_t parameter_bytes() const noexcept;
    [[nodiscard]] std::size_t position_bytes() const noexcept {
        return positions_.size() * sizeof(float);
    }
    // Each output is one point forecast at that row's origin. Benchmark uses last.
    [[nodiscard]] Error forward(std::span<const float> input, std::size_t tokens,
                                std::span<float> output,
                                ForecastWorkspace &workspace) const noexcept;

  private:
    ForecastConfig config_;
    Dense input_;
    TransformerBlock block_;
    Dense head_;
    AlignedBuffer scaler_, final_norm_, positions_;
};
} // namespace stealth::numeric
