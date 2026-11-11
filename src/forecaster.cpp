#include "stealth/forecaster.hpp"
#include "fp_policy.hpp"
#include "sequence_detail.hpp"
#include <algorithm>

namespace stealth::numeric {
namespace {
ForecastConfig checked(ForecastConfig c) {
    sequence_detail::dimensions(c.context, c.width, c.heads);
    if (c.features == 0 || c.hidden == 0 || c.horizon == 0 || c.target >= c.features ||
        !std::isfinite(c.epsilon) || c.epsilon <= 0 ||
        (c.output != ForecastOutput::level && c.output != ForecastOutput::correction))
        throw std::invalid_argument("Invalid forecast configuration");
    if (c.features > max_features || c.hidden > max_hidden || c.horizon > max_context)
        throw std::length_error("Forecast capacity exceeded");
    return c;
}
ForecastConfig checked_parameters(ForecastConfig c, std::span<const float> p) {
    if (p.size() != forecast_parameter_count(c) || !sequence_detail::finite(p))
        throw std::invalid_argument("Invalid forecast parameter count/value");
    for (float scale : p.subspan(c.features, c.features))
        if (scale <= 0)
            throw std::invalid_argument("Forecast scales must be positive");
    return c;
}
struct Cursor {
    std::span<const float> p;
    std::size_t offset;
    std::span<const float> take(std::size_t n) {
        auto result = p.subspan(offset, n);
        offset += n;
        return result;
    }
};
TransformerParameters block_parameters(ForecastConfig c, std::span<const float> p) {
    Cursor s{p, 2 * c.features + c.width * c.features + c.width};
    const auto w = c.width, h = c.hidden;
    auto scale = s.take(w), shift = s.take(w), qkv = s.take(3 * w * w), qkv_bias = s.take(3 * w);
    auto projection = s.take(w * w), projection_bias = s.take(w);
    auto ff_scale = s.take(w), ff_shift = s.take(w), up = s.take(h * w), up_bias = s.take(h);
    auto down = s.take(w * h), down_bias = s.take(w);
    return {scale,
            shift,
            qkv,
            qkv_bias,
            projection,
            projection_bias,
            {ff_scale, ff_shift, up, up_bias, down, down_bias}};
}
} // namespace
std::size_t forecast_parameter_count(ForecastConfig c) {
    c = checked(c);
    const auto w = c.width, h = c.hidden, f = c.features;
    return 4 * w * w + 2 * w * h + 13 * w + h + w * f + 2 * f + 1;
}
ForecastWorkspace::ForecastWorkspace(ForecastConfig c)
    : config_(checked(c)), storage_((2 * c.context + 2) * c.width + c.context + c.features + 1),
      block_(c.context, c.width, c.heads, c.hidden) {}
std::size_t ForecastWorkspace::bytes() const noexcept {
    return storage_.size() * sizeof(float) + block_.bytes();
}
ForecastModel::ForecastModel(ForecastConfig c, std::span<const float> p, DenseKernel kernel)
    : config_(checked_parameters(c, p)),
      input_(c.features, c.width, p.subspan(2 * c.features, c.width * c.features),
             p.subspan(2 * c.features + c.width * c.features, c.width), kernel),
      block_(c.width, c.heads, c.hidden, block_parameters(c, p), c.epsilon, kernel),
      head_(c.width, 1, p.subspan(p.size() - c.width - 1, c.width), p.last(1), kernel),
      scaler_(2 * c.features), final_norm_(2 * c.width), positions_(c.context * c.width) {
    std::copy_n(p.begin(), 2 * c.features, scaler_.span().begin());
    std::copy_n(p.end() - static_cast<std::ptrdiff_t>(3 * c.width + 1), 2 * c.width,
                final_norm_.span().begin());
    for (std::size_t t = 0; t < c.context; ++t)
        for (std::size_t channel = 0; channel < c.width; ++channel) {
            const auto exponent =
                static_cast<double>(2 * (channel / 2)) / static_cast<double>(c.width);
            const auto angle = static_cast<double>(t) / std::pow(10000.0, exponent);
            positions_.span()[t * c.width + channel] =
                static_cast<float>(channel % 2 == 0 ? std::sin(angle) : std::cos(angle));
        }
}
std::size_t ForecastModel::parameter_bytes() const noexcept {
    return input_.parameter_bytes() + block_.parameter_bytes() + head_.parameter_bytes() +
           (scaler_.size() + final_norm_.size()) * sizeof(float);
}
Error ForecastModel::forward(std::span<const float> input, std::size_t tokens,
                             std::span<float> output, ForecastWorkspace &ws,
                             SequenceOutput mode) const noexcept {
    const auto c = config_;
    if (scaler_.size() != 2 * c.features || final_norm_.size() != 2 * c.width ||
        positions_.size() != c.context * c.width || ws.storage_.size() == 0)
        return Error::invalid_state;
    if ((mode != SequenceOutput::all && mode != SequenceOutput::last) || tokens == 0 ||
        tokens > c.context || ws.config_ != c || input.size() != tokens * c.features)
        return Error::shape_mismatch;
    const auto first_row = mode == SequenceOutput::last ? tokens - 1 : 0;
    if (output.size() != tokens - first_row)
        return Error::shape_mismatch;
    if (sequence_detail::overlaps(input, output))
        return Error::aliasing;
    if (!sequence_detail::finite(input))
        return Error::non_finite;
    auto storage = ws.storage_.span();
    const auto capacity = c.context * c.width;
    auto embedded = storage.subspan(0, tokens * c.width);
    auto transformed = storage.subspan(capacity, (tokens - first_row) * c.width);
    auto candidate = storage.subspan(2 * capacity, tokens);
    const auto offset = 2 * capacity + c.context;
    auto features = storage.subspan(offset, c.features);
    auto row = storage.subspan(offset + c.features, c.width);
    auto scratch = storage.subspan(offset + c.features + c.width, c.width);
    auto readout = storage.subspan(offset + c.features + 2 * c.width, 1);
    const auto means = scaler_.span().first(c.features);
    const auto scales = scaler_.span().subspan(c.features);
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t j = 0; j < c.features; ++j)
            if (!sequence_detail::checked_float(
                    (static_cast<double>(input[t * c.features + j]) - means[j]) / scales[j],
                    features[j]))
                return Error::numerical_error;
        auto embedding = embedded.subspan(t * c.width, c.width);
        auto status = input_.forward(features, embedding, scratch);
        if (status != Error::ok)
            return status;
        for (std::size_t j = 0; j < c.width; ++j)
            if (!sequence_detail::checked_float(static_cast<double>(embedding[j]) +
                                                    positions_.span()[t * c.width + j],
                                                embedding[j]))
                return Error::numerical_error;
    }
    auto status = block_.forward(embedded, tokens, transformed, ws.block_, mode);
    if (status != Error::ok)
        return status;
    for (std::size_t t = first_row; t < tokens; ++t) {
        status = layer_norm(transformed.subspan((t - first_row) * c.width, c.width),
                            final_norm_.span().first(c.width), final_norm_.span().subspan(c.width),
                            c.epsilon, row, scratch);
        if (status != Error::ok)
            return status;
        status = head_.forward(row, readout, scratch);
        if (status != Error::ok)
            return status;
        const auto anchor = c.output == ForecastOutput::correction
                                ? input[t * c.features + c.target]
                                : means[c.target];
        if (!sequence_detail::checked_float(
                static_cast<double>(readout[0]) * scales[c.target] + anchor, candidate[t]))
            return Error::numerical_error;
    }
    std::copy(candidate.begin() + static_cast<std::ptrdiff_t>(first_row), candidate.end(),
              output.begin());
    return Error::ok;
}
namespace {
std::size_t checked_batch(std::size_t batch) {
    if (batch == 0)
        throw std::invalid_argument("Batch capacity must be positive");
    if (batch > 1024)
        throw std::length_error("Batch capacity exceeded");
    return batch;
}
constexpr std::size_t forecast_batch_rows = 32;
} // namespace
ForecastBatchWorkspace::ForecastBatchWorkspace(ForecastConfig c, std::size_t batch_capacity)
    : config_(checked(c)), capacity_(checked_batch(batch_capacity)),
      tile_(std::min(std::size_t{4}, capacity_)),
      storage_(2 * tile_ * c.context * c.width + capacity_ * c.context +
               forecast_batch_rows * (c.features + 2 * c.width + 1)),
      block_(c.context, c.width, c.heads, c.hidden, tile_) {}
std::size_t ForecastBatchWorkspace::bytes() const noexcept {
    return storage_.size() * sizeof(float) + block_.bytes();
}
Error ForecastModel::forward_batch(std::span<const float> input, std::size_t batch,
                                   std::size_t tokens, std::span<float> output,
                                   ForecastBatchWorkspace &ws, SequenceOutput mode) const noexcept {
    const auto c = config_;
    if (scaler_.size() != 2 * c.features || final_norm_.size() != 2 * c.width ||
        positions_.size() != c.context * c.width || ws.storage_.size() == 0)
        return Error::invalid_state;
    if ((mode != SequenceOutput::all && mode != SequenceOutput::last) || batch == 0 ||
        batch > ws.capacity_ || tokens == 0 || tokens > c.context || ws.config_ != c ||
        input.size() > max_elements || input.size() != batch * tokens * c.features)
        return Error::shape_mismatch;
    const auto out_tokens = mode == SequenceOutput::last ? std::size_t{1} : tokens;
    if (output.size() != batch * out_tokens)
        return Error::shape_mismatch;
    if (sequence_detail::overlaps(input, output))
        return Error::aliasing;
    if (!sequence_detail::finite(input))
        return Error::non_finite;
    const auto capacity = ws.tile_ * c.context * c.width;
    auto storage = ws.storage_.span();
    auto embedded = storage.first(capacity);
    auto transformed = storage.subspan(capacity, capacity);
    auto candidate = storage.subspan(2 * capacity, batch * out_tokens);
    const auto offset = 2 * capacity + ws.capacity_ * c.context;
    auto features = storage.subspan(offset, forecast_batch_rows * c.features);
    auto normalized =
        storage.subspan(offset + forecast_batch_rows * c.features, forecast_batch_rows * c.width);
    auto scratch = storage.subspan(offset + forecast_batch_rows * (c.features + c.width),
                                   forecast_batch_rows * c.width);
    auto readout = storage.subspan(offset + forecast_batch_rows * (c.features + 2 * c.width),
                                   forecast_batch_rows);
    const auto means = scaler_.span().first(c.features),
               scales = scaler_.span().subspan(c.features);
    for (std::size_t b = 0; b < batch; b += ws.tile_) {
        const auto windows = std::min(ws.tile_, batch - b);
        for (std::size_t t = 0; t < windows * tokens; t += forecast_batch_rows) {
            const auto rows = std::min(forecast_batch_rows, windows * tokens - t);
            for (std::size_t r = 0; r < rows; ++r)
                for (std::size_t j = 0; j < c.features; ++j)
                    if (!sequence_detail::checked_float(
                            (static_cast<double>(input[(b * tokens + t + r) * c.features + j]) -
                             means[j]) /
                                scales[j],
                            features[r * c.features + j]))
                        return Error::numerical_error;
            auto embedding = embedded.subspan(t * c.width, rows * c.width);
            const auto status =
                input_.forward_rows(features.first(rows * c.features), rows, embedding, scratch);
            if (status != Error::ok)
                return status;
            // Position depends on the row, never the channel. GCC 11 otherwise
            // emitted a 64-bit divide per channel in the nested expression.
            auto position = t % tokens;
            for (std::size_t r = 0; r < rows; ++r) {
                auto row = embedding.subspan(r * c.width, c.width);
                const auto positional = positions_.span().subspan(position * c.width, c.width);
                for (std::size_t j = 0; j < c.width; ++j)
                    if (!sequence_detail::checked_float(static_cast<double>(row[j]) + positional[j],
                                                        row[j]))
                        return Error::numerical_error;
                if (++position == tokens)
                    position = 0;
            }
        }
        auto status = block_.forward_batch(
            embedded.first(windows * tokens * c.width), windows, tokens,
            transformed.first(windows * out_tokens * c.width), ws.block_, mode);
        if (status != Error::ok)
            return status;
        for (std::size_t t = 0; t < windows * out_tokens; t += forecast_batch_rows) {
            const auto rows = std::min(forecast_batch_rows, windows * out_tokens - t);
            status = layer_norm_rows(transformed.subspan(t * c.width, rows * c.width), rows,
                                     final_norm_.span().first(c.width),
                                     final_norm_.span().subspan(c.width), c.epsilon,
                                     normalized.first(rows * c.width), scratch);
            if (status != Error::ok)
                return status;
            status = head_.forward_rows(normalized.first(rows * c.width), rows, readout.first(rows),
                                        scratch);
            if (status != Error::ok)
                return status;
            for (std::size_t r = 0; r < rows; ++r) {
                const auto origin =
                    b * tokens + (mode == SequenceOutput::last ? (t + r + 1) * tokens - 1 : t + r);
                const auto anchor = c.output == ForecastOutput::correction
                                        ? input[origin * c.features + c.target]
                                        : means[c.target];
                if (!sequence_detail::checked_float(
                        static_cast<double>(readout[r]) * scales[c.target] + anchor,
                        candidate[b * out_tokens + t + r]))
                    return Error::numerical_error;
            }
        }
    }
    std::copy(candidate.begin(), candidate.end(), output.begin());
    return Error::ok;
}
} // namespace stealth::numeric
