#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace stealth::temporal {
inline constexpr std::size_t max_signal_features = 256;
inline constexpr std::size_t max_signal_deliveries = 100000;

struct SignalLimits {
    std::int64_t max_observation_age_ns;
    std::int64_t max_publication_age_ns;
};
struct SignalRecord {
    std::size_t channel;
    std::uint64_t event;
    std::int64_t revision;
    std::int64_t effective_at_ns;
    std::int64_t available_at_ns;
    std::int64_t received_at_ns;
    std::optional<double> value;
    bool operator==(const SignalRecord &) const = default;
};
enum class SignalState { missing, withdrawn, stale, valid };
enum class SnapshotError { ok, invalid_time, output_size };
struct SignalFeature {
    SignalState state = SignalState::missing;
    std::optional<double> value;
    std::optional<SignalRecord> observation;
    // Zero when observation is absent; inspect observation/state first.
    std::int64_t observation_age_ns = 0;
    std::int64_t publication_age_ns = 0;
    std::int64_t receipt_age_ns = 0;
    bool operator==(const SignalFeature &) const = default;
};

// Immutable finite replay index. Numeric channel/event mapping is caller-owned.
// Construction copies, validates and indexes all deliveries, or throws.
// Concurrent const snapshots are safe with separate caller-owned output spans.
class SignalIndex {
  public:
    SignalIndex(std::span<const SignalLimits> limits, std::span<const SignalRecord> records);
    [[nodiscard]] std::size_t feature_count() const noexcept { return limits_.size(); }
    [[nodiscard]] std::size_t record_count() const noexcept { return record_count_; }
    [[nodiscard]] std::size_t index_entry_count() const noexcept { return index_entry_count_; }
    // Strict receipt < decision. No allocation, including rejected calls.
    // Invalid time or output size leaves output unchanged.
    [[nodiscard]] SnapshotError snapshot(std::int64_t decision_at_ns,
                                         std::span<SignalFeature> output) const noexcept;

  private:
    std::vector<SignalLimits> limits_;
    std::vector<std::vector<SignalRecord>> changes_;
    std::size_t record_count_ = 0;
    std::size_t index_entry_count_ = 0;
};
} // namespace stealth::temporal
