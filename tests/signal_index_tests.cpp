#include "stealth/signal_index.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <random>
#include <thread>
#include <vector>

using namespace stealth::temporal;
using I = std::int64_t;
namespace {
constexpr I max_time = std::numeric_limits<I>::max();
SignalRecord row(std::size_t channel, std::uint64_t event, I revision, I effective,
                 I published, I received, std::optional<double> value) {
    return {channel, event, revision, effective, published, received, value};
}
std::vector<SignalFeature> snapshot(const SignalIndex &index, I time) {
    std::vector<SignalFeature> out(index.feature_count());
    CHECK(index.snapshot(time, out) == SnapshotError::ok);
    return out;
}
void hand() {
    const std::array<SignalLimits, 2> limits{{{100, 30}, {200, 200}}};
    std::vector<SignalRecord> records{
        row(0, 1, 0, 10, 10, 15, 1), row(0, 1, 1, 10, 20, 40, 2),
        row(0, 1, 2, 10, 21, 25, 3), // newer revision arrives first
        row(0, 2, 0, 30, 30, 35, 4),
        row(0, 1, 3, 10, 50, 50, 9), // older event correction
        row(0, 2, 1, 30, 60, 60, std::nullopt),
        row(0, 2, 2, 30, 70, 70, 6),
        row(1, 1, 0, 0, 0, 1, -0.0),
        row(0, 1, 0, 10, 10, 18, 1)};
    const SignalIndex index(limits, records);
    CHECK(index.record_count() == 8);
    CHECK(index.index_entry_count() <= index.record_count());
    CHECK(snapshot(index, 0)[0].state == SignalState::missing);
    CHECK(snapshot(index, 15)[0].state == SignalState::missing);
    auto a = snapshot(index, 16)[0];
    CHECK(a.value == 1 && a.observation->received_at_ns == 15);
    CHECK(a.observation_age_ns == 6 && a.publication_age_ns == 6 && a.receipt_age_ns == 1);
    CHECK(snapshot(index, 25)[0].value == 1);
    CHECK(snapshot(index, 26)[0].value == 3);
    CHECK(snapshot(index, 35)[0].value == 3);
    CHECK(snapshot(index, 36)[0].value == 4);
    CHECK(snapshot(index, 51)[0].value == 4);
    CHECK(snapshot(index, 60)[0].value == 4); // age equals limit
    CHECK(snapshot(index, 61)[0].state == SignalState::withdrawn);
    CHECK(snapshot(index, 71)[0].value == 6);
    CHECK(snapshot(index, 101)[0].state == SignalState::stale);
    CHECK(!snapshot(index, 101)[0].value);
    CHECK(snapshot(index, 101)[0].observation->value == 6);
    CHECK(std::signbit(*snapshot(index, 2)[1].value));
    // Older revision arriving late must not undo the received newer revision.
    const std::array late{records[0], records[1], records[2]};
    CHECK(snapshot(SignalIndex(limits, late), 41)[0].value == 3);
    // Same receipt: all eligible revisions are considered together.
    const std::array tie{row(0, 2, 1, 20, 21, 30, 7), row(0, 2, 0, 20, 20, 30, 8)};
    CHECK(snapshot(SignalIndex(limits, tie), 31)[0].value == 7);
}
void failures() {
    const std::array<SignalLimits, 1> limits{{{100, 100}}};
    const auto original = row(0, 1, 0, 1, 2, 3, 1);
    auto rejects = [&](std::vector<SignalRecord> records) {
        test::throws<std::invalid_argument>([&] { SignalIndex bad(limits, records); });
    };
    for (int field = 0; field < 8; ++field) {
        auto r = original;
        switch (field) {
        case 0: r.channel = 1; break;
        case 1: r.revision = -1; break;
        case 2: r.effective_at_ns = -1; break;
        case 3: r.available_at_ns = 0; break;
        case 4: r.received_at_ns = 1; break;
        case 5: r.value = std::numeric_limits<double>::infinity(); break;
        case 6: r.value = std::numeric_limits<double>::quiet_NaN(); break;
        default: r.received_at_ns = -1; break;
        }
        rejects({r});
    }
    for (int field = 0; field < 5; ++field) {
        auto r = original;
        switch (field) {
        case 0: r.value = 2; break; // duplicate conflict
        case 1: r.available_at_ns = 3; break;
        case 2: r.revision = 1; r.effective_at_ns = 2; break; // event moved
        case 3: r.event = 2; break; // ambiguous point
        default: r.revision = 1; r.available_at_ns = 1; break;
        }
        rejects({original, r});
    }
    auto zero = original; zero.value = 0.0;
    auto negative_zero = zero; negative_zero.value = -0.0;
    rejects({zero, negative_zero});
    auto withdrawal = original; withdrawal.value = std::nullopt;
    rejects({original, withdrawal});
    const std::array<SignalLimits, 1> invalid{{{-1, 0}}};
    test::throws<std::invalid_argument>([&] { SignalIndex bad(invalid, {}); });
    test::throws<std::invalid_argument>([] { SignalIndex bad({}, {}); });
    std::vector<SignalLimits> excess(257, {0, 0});
    test::throws<std::invalid_argument>([&] { SignalIndex bad(excess, {}); });
    const SignalIndex index(limits, {&original, 1});
    auto out = snapshot(index, 4);
    const auto before = out;
    CHECK(index.snapshot(-1, out) == SnapshotError::invalid_time);
    CHECK(out == before);
    out.push_back(before[0]);
    const auto too_large = out;
    CHECK(index.snapshot(4, out) == SnapshotError::output_size);
    CHECK(out == too_large);
    CHECK(index.snapshot(4, {}) == SnapshotError::output_size);
}
SignalFeature scan(const std::vector<SignalRecord> &records, std::size_t channel,
                   SignalLimits limits, I decision) {
    std::optional<SignalRecord> selected;
    for (const auto &r : records) {
        if (r.channel != channel || r.received_at_ns >= decision)
            continue;
        if (!selected || r.effective_at_ns > selected->effective_at_ns ||
            (r.effective_at_ns == selected->effective_at_ns && r.revision > selected->revision) ||
            (r.event == selected->event && r.revision == selected->revision &&
             r.received_at_ns < selected->received_at_ns))
            selected = r;
    }
    SignalFeature out{};
    if (selected) {
        out.observation = selected;
        out.observation_age_ns = decision - selected->effective_at_ns;
        out.publication_age_ns = decision - selected->available_at_ns;
        out.receipt_age_ns = decision - selected->received_at_ns;
        if (!selected->value)
            out.state = SignalState::withdrawn;
        else if (out.observation_age_ns > limits.max_observation_age_ns ||
                 out.publication_age_ns > limits.max_publication_age_ns)
            out.state = SignalState::stale;
        else {
            out.state = SignalState::valid;
            out.value = selected->value;
        }
    }
    return out;
}
void generated() {
    std::mt19937 random(721);
    const std::array<SignalLimits, 3> limits{{{25, 20}, {500, 50}, {100, 100}}};
    std::vector<SignalRecord> records;
    for (std::size_t channel = 0; channel < limits.size(); ++channel)
        for (I event = 0; event < 60; ++event)
            for (I revision = 0; revision < 3; ++revision) {
                const I published = event * 10 + revision * 2;
                const I received = published + static_cast<I>(random() % 100);
                auto r = row(channel, static_cast<std::uint64_t>(event), revision,
                             event * 10, published, received, static_cast<double>(event + revision));
                if (random() % 11 == 0)
                    r.value = std::nullopt;
                records.push_back(r);
                if (random() % 5 == 0) {
                    r.received_at_ns += 5;
                    records.push_back(r);
                }
            }
    std::shuffle(records.begin(), records.end(), random);
    const SignalIndex a(limits, records);
    std::reverse(records.begin(), records.end());
    const SignalIndex restarted(limits, records);
    CHECK(a.index_entry_count() <= a.record_count());
    for (I t = 800; t >= 0; --t) {
        const auto actual = snapshot(a, t);
        CHECK(actual == snapshot(restarted, t));
        for (std::size_t c = 0; c < limits.size(); ++c)
            CHECK(actual[c] == scan(records, c, limits[c], t));
    }
}
void boundaries() {
    const std::array<SignalLimits, 1> limits{{{max_time, max_time}}};
    std::vector<SignalRecord> records{row(0, 0, max_time, 0, 0, 0, 2),
                                    row(0, 1, max_time, max_time, max_time, max_time, 3)};
    const SignalIndex index(limits, records);
    CHECK(snapshot(index, max_time)[0].value == 2);
    CHECK(snapshot(index, max_time)[0].observation_age_ns == max_time);
    const std::array<SignalLimits, 1> zero{{{0, 0}}};
    CHECK(snapshot(SignalIndex(zero, records), 1)[0].state == SignalState::stale);
    const auto repeated = records[0]; // Copy before assign invalidates vector references.
    records.assign(100000, repeated);
    CHECK(SignalIndex(limits, records).record_count() == 1);
    records.push_back(repeated);
    test::throws<std::invalid_argument>([&] { SignalIndex bad(limits, records); });
    records.clear();
    for (I i = 0; i < 100000; ++i)
        records.push_back(row(0, static_cast<std::uint64_t>(i), 0, i, i, i,
                              static_cast<double>(i)));
    const SignalIndex maximum(limits, records);
    CHECK(maximum.record_count() == 100000);
    CHECK(maximum.index_entry_count() == 100000);
    CHECK(snapshot(maximum, 100000)[0].value == 99999.0);
    const std::vector<SignalLimits> many(256, {0, 0});
    CHECK(snapshot(SignalIndex(many, {}), 0).size() == 256);
}
void future() {
    const std::array<SignalLimits, 1> limits{{{1000, 1000}}};
    std::vector<SignalRecord> records{row(0, 1, 0, 10, 10, 10, 1)};
    const SignalIndex before(limits, records);
    records.push_back(row(0, 1, 1, 10, 20, 100, 100));
    records.push_back(row(0, 2, 0, 90, 90, 100, 200));
    const SignalIndex after(limits, records);
    for (I t = 0; t <= 100; ++t)
        CHECK(snapshot(before, t) == snapshot(after, t));
    CHECK(snapshot(after, 101)[0].value == 200);
}
void concurrent() {
    const std::array<SignalLimits, 1> limits{{{1000, 1000}}};
    const std::array records{row(0, 1, 0, 0, 0, 1, 2), row(0, 2, 0, 10, 10, 11, 3)};
    const SignalIndex index(limits, records);
    std::atomic<bool> passed{true};
    std::vector<std::thread> threads;
    for (int n = 0; n < 4; ++n)
        threads.emplace_back([&] {
            std::array<SignalFeature, 1> out;
            for (int i = 0; i < 2000; ++i)
                if (index.snapshot(i % 2 == 0 ? 2 : 12, out) != SnapshotError::ok ||
                    out[0].value != (i % 2 == 0 ? 2.0 : 3.0))
                    passed.store(false);
        });
    for (auto &thread : threads)
        thread.join();
    CHECK(passed.load());
}
}
int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Expected group");
        const std::string group(argv[1]);
        if (group == "hand") hand();
        else if (group == "failures") failures();
        else if (group == "generated") generated();
        else if (group == "boundaries") boundaries();
        else if (group == "future") future();
        else if (group == "concurrent") concurrent();
        else throw std::invalid_argument("Unknown group");
        std::cout << test::checks << " checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
