#include "fixtures/attention.hpp"
#include "fixtures/feed_forward.hpp"
#include "fixtures/forecast.hpp"
#include "stealth/feed_forward.hpp"
#include "stealth/model_file.hpp"
#include "stealth/transformer_block.hpp"
#include "test_support.hpp"

#include <array>
#include <cstdlib>
#include <limits>
#include <new>
#ifdef _WIN32
#include <malloc.h>
#endif

namespace {
thread_local bool tracking = false;
thread_local std::size_t allocations = 0;
thread_local bool inject_failure = false;
thread_local std::size_t fail_index = 0, attempt = 0;
void maybe_fail() {
    if (inject_failure && attempt++ == fail_index)
        throw std::bad_alloc();
}
void *allocate(std::size_t bytes) {
    if (tracking)
        ++allocations;
    maybe_fail();
    if (void *address = std::malloc(bytes == 0 ? 1 : bytes))
        return address;
    throw std::bad_alloc();
}
void *allocate_aligned(std::size_t bytes, std::size_t alignment) {
    if (tracking)
        ++allocations;
    maybe_fail();
#ifdef _WIN32
    if (void *address = _aligned_malloc(bytes == 0 ? 1 : bytes, alignment))
        return address;
#else
    void *address = nullptr;
    if (posix_memalign(&address, alignment, bytes == 0 ? 1 : bytes) == 0)
        return address;
#endif
    throw std::bad_alloc();
}
void free_aligned(void *address) noexcept {
#ifdef _WIN32
    _aligned_free(address);
#else
    std::free(address);
#endif
}
} // namespace

void *operator new(std::size_t bytes) { return allocate(bytes); }
void *operator new[](std::size_t bytes) { return allocate(bytes); }
void operator delete(void *address) noexcept { std::free(address); }
void operator delete[](void *address) noexcept { std::free(address); }
void operator delete(void *address, std::size_t) noexcept { std::free(address); }
void operator delete[](void *address, std::size_t) noexcept { std::free(address); }
void *operator new(std::size_t bytes, std::align_val_t alignment) {
    return allocate_aligned(bytes, static_cast<std::size_t>(alignment));
}
void *operator new[](std::size_t bytes, std::align_val_t alignment) {
    return allocate_aligned(bytes, static_cast<std::size_t>(alignment));
}
void operator delete(void *address, std::align_val_t) noexcept { free_aligned(address); }
void operator delete[](void *address, std::align_val_t) noexcept { free_aligned(address); }
void operator delete(void *address, std::size_t, std::align_val_t) noexcept {
    free_aligned(address);
}
void operator delete[](void *address, std::size_t, std::align_val_t) noexcept {
    free_aligned(address);
}

int main() {
    using namespace stealth::numeric;
    try {
        // Positive control proves the interception is live for both allocation families.
        tracking = true;
        void *plain = ::operator new(17);
        void *aligned = ::operator new[](65, std::align_val_t{64});
        ::operator delete(plain);
        ::operator delete[](aligned, std::align_val_t{64});
        tracking = false;
        CHECK(allocations == 2);
        for (auto kernel : {DenseKernel::reference, DenseKernel::packed}) {
            FeedForward model(fixture::width, fixture::hidden,
                              {fixture::scale, fixture::shift, fixture::up_weights,
                               fixture::up_bias, fixture::down_weights, fixture::down_bias},
                              fixture::epsilon, kernel);
            FeedForwardWorkspace workspace(fixture::width, fixture::hidden);
            std::array<float, fixture::width> output{}, probabilities{}, scratch{},
                bad = fixture::input;
            bad[0] = std::numeric_limits<float>::infinity();
            CHECK(model.forward(fixture::input, output, workspace) == Error::ok);
            bool failed = false;
            allocations = 0;
            tracking = true;
            for (int iteration = 0; iteration < 10000; ++iteration) {
                if (model.forward(fixture::input, output, workspace) != Error::ok ||
                    softmax(output, output.size(), probabilities, scratch) != Error::ok ||
                    model.forward(bad, output, workspace) != Error::non_finite ||
                    model.forward(fixture::input, std::span(output).first(3), workspace) !=
                        Error::shape_mismatch) {
                    failed = true;
                    break;
                }
            }
            tracking = false;
            CHECK(!failed);
            CHECK(allocations == 0);
            for (std::size_t i = 0; i < output.size(); ++i)
                test::near(output[i], fixture::expected[i], 2e-5, 5e-5);
        }
        namespace f = attention_fixture;
        for (auto kernel : {DenseKernel::reference, DenseKernel::packed}) {
            TransformerBlock model(f::width, f::heads, f::hidden,
                                   {f::attn_scale,
                                    f::attn_shift,
                                    f::qkv_weights,
                                    f::qkv_bias,
                                    f::projection_weights,
                                    f::projection_bias,
                                    {f::ff_scale, f::ff_shift, f::up_weights, f::up_bias,
                                     f::down_weights, f::down_bias}},
                                   f::epsilon, kernel);
            TransformerWorkspace workspace(f::tokens, f::width, f::heads, f::hidden);
            AttentionWorkspace attention(f::tokens, f::width, f::heads);
            std::array<float, f::tokens * f::width> output{}, context{}, bad = f::input;
            bad.back() = std::numeric_limits<float>::infinity();
            CHECK(model.forward(f::input, f::tokens, output, workspace) == Error::ok);
            allocations = 0;
            bool failed = false;
            tracking = true;
            for (int i = 0; i < 2000; ++i) {
                if (model.forward(f::input, f::tokens, output, workspace) != Error::ok ||
                    causal_attention(f::query, f::key, f::value, f::tokens, context, attention) !=
                        Error::ok ||
                    model.forward(bad, f::tokens, output, workspace) != Error::non_finite ||
                    model.forward(f::input, 0, output, workspace) != Error::shape_mismatch) {
                    failed = true;
                    break;
                }
            }
            tracking = false;
            CHECK(!failed);
            CHECK(allocations == 0);
            for (std::size_t i = 0; i < output.size(); ++i)
                test::near(output[i], f::expected[i], 3e-5, 1e-4);
        }
        namespace ff = forecast_fixture;
        for (auto kernel : {DenseKernel::reference, DenseKernel::packed}) {
            auto model = load_forecaster(std::as_bytes(std::span(ff::artifact)), kernel);
            ForecastWorkspace workspace(model->config());
            std::array<float, ff::tokens> output{};
            auto bad = ff::input;
            bad.back() = std::numeric_limits<float>::infinity();
            allocations = 0;
            tracking = true;
            bool failed = false;
            for (int i = 0; i < 2000; ++i) {
                if (model->forward(ff::input, ff::tokens, output, workspace) != Error::ok ||
                    model->forward(bad, ff::tokens, output, workspace) != Error::non_finite ||
                    model->forward(ff::input, 0, output, workspace) != Error::shape_mismatch) {
                    failed = true;
                    break;
                }
            }
            tracking = false;
            CHECK(!failed);
            CHECK(allocations == 0);
            for (std::size_t i = 0; i < output.size(); ++i)
                test::near(output[i], ff::expected[i], 5e-5, 2e-4);
            // Inject bad_alloc at every intercepted allocation of a valid cold load.
            allocations = 0;
            tracking = true;
            auto candidate = load_forecaster(std::as_bytes(std::span(ff::artifact)), kernel);
            tracking = false;
            const auto cold_allocations = allocations;
            CHECK(cold_allocations > 1);
            candidate.reset();
            const auto *before = model.get();
            for (std::size_t failure = 0; failure < cold_allocations; ++failure) {
                fail_index = failure;
                attempt = 0;
                inject_failure = true;
                bool caught = false;
                try {
                    model = load_forecaster(std::as_bytes(std::span(ff::artifact)), kernel);
                } catch (const std::bad_alloc &) {
                    caught = true;
                }
                inject_failure = false;
                CHECK(caught);
                CHECK(model.get() == before);
                CHECK(model->forward(ff::input, ff::tokens, output, workspace) == Error::ok);
                for (std::size_t i = 0; i < output.size(); ++i)
                    test::near(output[i], ff::expected[i], 5e-5, 2e-4);
            }
            std::cout << "Preserved active model across " << cold_allocations
                      << " injected load allocation failures\n";
        }
        std::cout << "No intercepted C++ heap allocations in 4,000 forecaster/error-path cycles\n";
        std::cout << "No intercepted C++ heap allocations in 4,000 "
                     "transformer/attention/error-path cycles\n";
        std::cout
            << "No intercepted C++ heap allocations in 20,000 block/softmax/error-path cycles\n";
        return 0;
    } catch (const std::exception &error) {
        tracking = false;
        inject_failure = false;
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
