#include "stealth/signal_index.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace stealth::temporal {
namespace {
bool same_payload(const SignalRecord &a, const SignalRecord &b) noexcept {
    return a.effective_at_ns == b.effective_at_ns && a.available_at_ns == b.available_at_ns &&
           a.value.has_value() == b.value.has_value() &&
           (!a.value || std::bit_cast<std::uint64_t>(*a.value) ==
                            std::bit_cast<std::uint64_t>(*b.value));
}
bool newer(const SignalRecord &a, const SignalRecord &b) noexcept {
    return std::tie(a.effective_at_ns, a.revision) > std::tie(b.effective_at_ns, b.revision);
}
}
SignalIndex::SignalIndex(std::span<const SignalLimits> limits,
                         std::span<const SignalRecord> deliveries) {
    if (limits.empty() || limits.size() > max_signal_features ||
        deliveries.size() > max_signal_deliveries)
        throw std::invalid_argument("Signal index capacity exceeded or empty schema");
    for (const auto &limit : limits)
        if (limit.max_observation_age_ns < 0 || limit.max_publication_age_ns < 0)
            throw std::invalid_argument("Negative signal age limit");
    for (const auto &r : deliveries)
        if (r.channel >= limits.size() || r.revision < 0 || r.effective_at_ns < 0 ||
            r.effective_at_ns > r.available_at_ns || r.available_at_ns > r.received_at_ns ||
            (r.value && !std::isfinite(*r.value)))
            throw std::invalid_argument("Invalid signal record");

    // Identity/revision order checks lineage; receipt order keeps earliest duplicates.
    std::vector<SignalRecord> records(deliveries.begin(), deliveries.end());
    std::sort(records.begin(), records.end(), [](const auto &a, const auto &b) {
        return std::tie(a.channel, a.event, a.revision, a.received_at_ns) <
               std::tie(b.channel, b.event, b.revision, b.received_at_ns);
    });
    std::size_t unique = 0;
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto r = records[i];
        if (unique != 0) {
            const auto &previous = records[unique - 1];
            if (previous.channel == r.channel && previous.event == r.event) {
                if (previous.effective_at_ns != r.effective_at_ns ||
                    previous.available_at_ns > r.available_at_ns)
                    throw std::invalid_argument("Inconsistent signal revision lineage");
                if (previous.revision == r.revision) {
                    if (!same_payload(previous, r))
                        throw std::invalid_argument("Conflicting signal duplicate");
                    continue;
                }
            }
        }
        records[unique++] = r;
    }
    records.resize(unique);
    record_count_ = unique;
    std::sort(records.begin(), records.end(), [](const auto &a, const auto &b) {
        return std::tie(a.channel, a.effective_at_ns, a.revision) <
               std::tie(b.channel, b.effective_at_ns, b.revision);
    });
    for (std::size_t i = 1; i < records.size(); ++i)
        if (records[i - 1].channel == records[i].channel &&
            records[i - 1].effective_at_ns == records[i].effective_at_ns &&
            records[i - 1].event != records[i].event)
            throw std::invalid_argument("Ambiguous signal event identity");

    limits_.assign(limits.begin(), limits.end());
    changes_.resize(limits.size());
    // Reserve at most one slot per unique record; queries never grow these vectors.
    std::vector<std::size_t> counts(limits.size(), 0);
    for (const auto &r : records)
        ++counts[r.channel];
    for (std::size_t c = 0; c < counts.size(); ++c)
        changes_[c].reserve(counts[c]);
    std::sort(records.begin(), records.end(), [](const auto &a, const auto &b) {
        return std::tie(a.received_at_ns, a.channel, a.effective_at_ns, a.revision) <
               std::tie(b.received_at_ns, b.channel, b.effective_at_ns, b.revision);
    });
    for (const auto &r : records) {
        auto &changes = changes_[r.channel];
        if (changes.empty() || newer(r, changes.back())) {
            if (!changes.empty() && changes.back().received_at_ns == r.received_at_ns)
                changes.back() = r;
            else {
                changes.push_back(r);
                ++index_entry_count_;
            }
        }
    }
}

SnapshotError SignalIndex::snapshot(std::int64_t decision,
                                    std::span<SignalFeature> output) const noexcept {
    if (decision < 0)
        return SnapshotError::invalid_time;
    if (output.size() != limits_.size())
        return SnapshotError::output_size;
    for (std::size_t c = 0; c < limits_.size(); ++c) {
        SignalFeature feature{};
        const auto &changes = changes_[c];
        auto next = std::lower_bound(changes.begin(), changes.end(), decision,
                                     [](const auto &r, std::int64_t t) {
                                         return r.received_at_ns < t;
                                     });
        if (next != changes.begin()) {
            const auto &r = *--next;
            feature.observation = r;
            // Nonnegative ordered clocks make every subtraction int64-safe.
            feature.observation_age_ns = decision - r.effective_at_ns;
            feature.publication_age_ns = decision - r.available_at_ns;
            feature.receipt_age_ns = decision - r.received_at_ns;
            if (!r.value)
                feature.state = SignalState::withdrawn;
            else if (feature.observation_age_ns > limits_[c].max_observation_age_ns ||
                     feature.publication_age_ns > limits_[c].max_publication_age_ns)
                feature.state = SignalState::stale;
            else {
                feature.state = SignalState::valid;
                feature.value = r.value;
            }
        }
        output[c] = feature;
    }
    return SnapshotError::ok;
}
} // namespace stealth::temporal
