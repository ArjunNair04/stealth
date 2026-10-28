#pragma once
#include "stealth/forecaster.hpp"
#include <memory>

namespace stealth::numeric {
// One writer per stream. Shared model stays alive and immutable for the stream lifetime.
// Each accepted row advances a bounded raw window; positions restart at zero.
class WindowStream {
  public:
    explicit WindowStream(std::shared_ptr<const ForecastModel> model);
    WindowStream(const WindowStream &) = delete;
    WindowStream &operator=(const WindowStream &) = delete;
    WindowStream(WindowStream &&) noexcept = default;
    WindowStream &operator=(WindowStream &&) noexcept = default;
    // Output has one element. It is untouched on failure and until the window is full.
    // Failure preserves the accepted history. Scratch contents are disposable.
    [[nodiscard]] Error push(std::span<const float> row, std::span<float> output) noexcept;
    void reset() noexcept { count_ = 0; }
    [[nodiscard]] std::size_t size() const noexcept { return model_ ? count_ : 0; }
    [[nodiscard]] bool ready() const noexcept {
        return model_ && count_ == model_->config().context;
    }
    [[nodiscard]] std::size_t bytes() const noexcept;

  private:
    std::shared_ptr<const ForecastModel> model_;
    ForecastWorkspace workspace_;
    AlignedBuffer storage_;
    std::size_t count_ = 0;
};
} // namespace stealth::numeric
