#include "fixtures/forecast.hpp"
#include "stealth/window_stream.hpp"
#include "test_support.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <string_view>
#include <thread>
#include <vector>
using namespace stealth::numeric;
namespace f = forecast_fixture;
namespace {
ForecastConfig config() {
    return {f::features, f::width, f::heads, f::hidden, f::tokens, f::horizon, f::target};
}
auto model() { return std::make_shared<const ForecastModel>(config(), f::parameters); }
auto row(std::size_t i) {
    return std::span(f::input).subspan((i % f::tokens) * f::features, f::features);
}
void rolling() {
    auto m = model();
    WindowStream stream(m);
    ForecastWorkspace ws(config());
    std::vector<float> history;
    std::array<float, f::tokens> full{};
    std::array<float, 1> out{123};
    for (std::size_t i = 0; i < 2000; ++i) {
        if (history.size() == f::input.size())
            history.erase(history.begin(), history.begin() + f::features);
        history.insert(history.end(), row(i).begin(), row(i).end());
        CHECK(stream.push(row(i), out) == Error::ok);
        const auto n = std::min(i + 1, f::tokens);
        CHECK(stream.size() == n);
        CHECK(m->forward(history, n, std::span(full).first(n), ws) == Error::ok);
        CHECK(stream.ready() == (n == f::tokens));
        if (stream.ready())
            test::near(out[0], full[n - 1], 0, 0);
        else
            test::unchanged(out);
        if (i == f::tokens - 1)
            test::near(out[0], f::expected.back(), 5e-5, 2e-4);
    }
}
void reset() {
    auto m = model();
    WindowStream a(m), b(m);
    std::array<float, 1> x{123}, y{123};
    for (std::size_t episode = 0; episode < 100; ++episode) {
        a.reset();
        a.reset();
        CHECK(a.size() == 0);
        CHECK(!a.ready());
        x[0] = 123;
        WindowStream fresh(m);
        for (std::size_t i = 0; i < f::tokens + 3; ++i) {
            y[0] = x[0];
            CHECK(a.push(row(i), x) == Error::ok);
            CHECK(fresh.push(row(i), y) == Error::ok);
            test::near(x[0], y[0], 0, 0);
            std::array<float, 1> unused{123};
            CHECK(b.push(row(i + 1), unused) == Error::ok);
        }
    }
}
void failures() {
    auto m = model();
    WindowStream a(m), b(m);
    std::array<float, 1> out{123}, reference{123};
    auto bad = f::input;
    bad[0] = std::numeric_limits<float>::quiet_NaN();
    for (std::size_t i = 0; i < f::tokens + 7; ++i) {
        auto count = a.size();
        auto old = out;
        CHECK(a.push(std::span(bad).first(f::features), out) == Error::non_finite);
        CHECK(a.push(row(i).first(1), out) == Error::shape_mismatch);
        CHECK(a.push(row(i), {}) == Error::shape_mismatch);
        auto alias = f::input;
        CHECK(a.push(std::span(alias).first(f::features), std::span(alias).first(1)) ==
              Error::aliasing);
        CHECK(alias == f::input);
        CHECK(a.size() == count);
        CHECK(out == old);
        CHECK(a.push(row(i), out) == Error::ok);
        CHECK(b.push(row(i), reference) == Error::ok);
        CHECK(out == reference);
    }
    auto p = f::parameters;
    p[f::features] = std::numeric_limits<float>::min();
    auto extreme = std::make_shared<const ForecastModel>(config(), p);
    WindowStream s(extreme);
    std::array<float, f::features> good{};
    std::copy_n(f::input.begin(), f::features, good.begin());
    good[0] = p[0];
    std::array<float, 1> y{123};
    CHECK(s.push(good, y) == Error::ok);
    CHECK(s.size() == 1);
    auto huge = good;
    huge[0] = std::numeric_limits<float>::max();
    CHECK(s.push(huge, y) == Error::numerical_error);
    CHECK(s.size() == 1);
    test::unchanged(y);
    CHECK(s.push(good, y) == Error::ok);
    CHECK(s.size() == 2);
    test::throws<std::invalid_argument>([] { WindowStream invalid(nullptr); });
}
void lifetime() {
    auto m = model();
    std::weak_ptr<const ForecastModel> weak = m;
    WindowStream a(m);
    m.reset();
    CHECK(!weak.expired());
    std::array<float, 1> out{123};
    CHECK(a.push(row(0), out) == Error::ok);
    auto b = std::move(a);
    CHECK(a.push(row(1), out) == Error::invalid_state);
    test::unchanged(out);
    CHECK(a.size() == 0);
    CHECK(!a.ready());
    a.reset();
    for (std::size_t i = 1; i < f::tokens; ++i)
        CHECK(b.push(row(i), out) == Error::ok);
    test::near(out[0], f::expected.back(), 5e-5, 2e-4);
}
void concurrent() {
    auto m = model();
    std::atomic<bool> good{true};
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 4; ++worker)
        workers.emplace_back([m, &good] {
            try {
                WindowStream s(m);
                std::array<float, 1> out{123};
                for (int episode = 0; episode < 100; ++episode) {
                    s.reset();
                    for (std::size_t i = 0; i < f::tokens; ++i)
                        if (s.push(row(i), out) != Error::ok)
                            good = false;
                    if (std::abs(out[0] - f::expected.back()) > 5e-5F)
                        good = false;
                }
            } catch (...) {
                good = false;
            }
        });
    for (auto &worker : workers)
        worker.join();
    CHECK(good.load());
}
void boundaries() {
    for (std::size_t context : {std::size_t{1}, max_context}) {
        ForecastConfig c{1, 1, 1, 1, context, 4, 0};
        std::vector<float> p(forecast_parameter_count(c), 0);
        p[0] = 20;
        p[1] = 2;
        p.back() = 3;
        auto m = std::make_shared<const ForecastModel>(c, p);
        WindowStream s(m);
        std::array<float, 1> x{7}, y{123};
        for (std::size_t i = 0; i < context + 1; ++i) {
            CHECK(s.push(x, y) == Error::ok);
            if (i + 1 < context)
                test::unchanged(y);
            else
                test::near(y[0], 26, 0, 0);
        }
        CHECK(s.size() == context);
        CHECK(s.bytes() > context * sizeof(float));
    }
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("Expected test group");
        const std::string_view group = argv[1];
        if (group == "rolling")
            rolling();
        else if (group == "reset")
            reset();
        else if (group == "failures")
            failures();
        else if (group == "lifetime")
            lifetime();
        else if (group == "concurrent")
            concurrent();
        else if (group == "boundaries")
            boundaries();
        else
            throw std::runtime_error("Unknown group");
        std::cout << group << ": " << test::checks << " checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
