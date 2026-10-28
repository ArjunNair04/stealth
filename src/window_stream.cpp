#include "stealth/window_stream.hpp"
#include "sequence_detail.hpp"
#include <algorithm>
#include <stdexcept>
namespace stealth::numeric {
namespace {
ForecastConfig checked_model(const std::shared_ptr<const ForecastModel> &model) {
    if (!model)
        throw std::invalid_argument("Stream requires a model");
    return model->config();
}
} // namespace
WindowStream::WindowStream(std::shared_ptr<const ForecastModel> model)
    : model_(std::move(model)), workspace_(checked_model(model_)),
      storage_(2 * model_->config().context * model_->config().features +
               model_->config().context) {}
std::size_t WindowStream::bytes() const noexcept {
    return storage_.size() * sizeof(float) + workspace_.bytes();
}
Error WindowStream::push(std::span<const float> row, std::span<float> output) noexcept {
    if (!model_ || storage_.size() == 0)
        return Error::invalid_state;
    const auto c = model_->config();
    if (row.size() != c.features || output.size() != 1)
        return Error::shape_mismatch;
    if (sequence_detail::overlaps(row, output))
        return Error::aliasing;
    if (!sequence_detail::finite(row))
        return Error::non_finite;
    const auto capacity = c.context * c.features;
    auto history = storage_.span().first(capacity);
    auto candidate = storage_.span().subspan(capacity, capacity);
    auto predictions = storage_.span().subspan(2 * capacity, c.context);
    const auto keep = count_ == c.context ? count_ - 1 : count_;
    const auto offset = count_ == c.context ? c.features : 0;
    std::copy_n(history.begin() + static_cast<std::ptrdiff_t>(offset), keep * c.features,
                candidate.begin());
    std::copy(row.begin(), row.end(),
              candidate.begin() + static_cast<std::ptrdiff_t>(keep * c.features));
    const auto next = keep + 1;
    const auto status = model_->forward(candidate.first(next * c.features), next,
                                        predictions.first(next), workspace_);
    if (status != Error::ok)
        return status;
    std::copy_n(candidate.begin(), next * c.features, history.begin());
    count_ = next;
    if (ready())
        output[0] = predictions[next - 1];
    return Error::ok;
}
} // namespace stealth::numeric
