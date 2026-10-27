#include "fixtures/feed_forward.hpp"
#include "stealth/feed_forward.hpp"
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
void *allocate(std::size_t bytes) {
    if (tracking)
        ++allocations;
    if (void *address = std::malloc(bytes == 0 ? 1 : bytes))
        return address;
    throw std::bad_alloc();
}
void *allocate_aligned(std::size_t bytes, std::size_t alignment) {
    if (tracking)
        ++allocations;
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
        std::cout
            << "No intercepted C++ heap allocations in 20,000 block/softmax/error-path cycles\n";
        return 0;
    } catch (const std::exception &error) {
        tracking = false;
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
