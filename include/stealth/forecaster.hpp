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
// Preallocated, move-only batch scratch. Activations are tiled over at most four
// windows; staged outputs cover batch_capacity so failures publish nothing.
class ForecastBatchWorkspace {
  public:
    ForecastBatchWorkspace(ForecastConfig config, std::size_t batch_capacity);
    [[nodiscard]] std::size_t bytes() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

  private:
    friend class ForecastModel;
    ForecastConfig config_;
    std::size_t capacity_, tile_;
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
    // All mode returns per-origin forecasts; last mode requires one output.
    // Latest-only skips unused diagnostic calculations in this one-block model.
    [[nodiscard]] Error forward(std::span<const float> input, std::size_t tokens,
                                std::span<float> output, ForecastWorkspace &workspace,
                                SequenceOutput mode = SequenceOutput::all) const noexcept;

    // Input [batch,tokens,features], output [batch,tokens] or [batch] in last mode.
    // No allocation; all windows are independent; output is unchanged on error.
    [[nodiscard]] Error forward_batch(std::span<const float> input, std::size_t batch,
                                      std::size_t tokens, std::span<float> output,
                                      ForecastBatchWorkspace &workspace,
                                      SequenceOutput mode = SequenceOutput::all) const noexcept;

  private:
    ForecastConfig config_;
    Dense input_;
    TransformerBlock block_;
    Dense head_;
    AlignedBuffer scaler_, final_norm_, positions_;
};
} // namespace stealth::numeric
