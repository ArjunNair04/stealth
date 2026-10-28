#include "fixtures/forecast.hpp"
#include "sha256.hpp"
#include "stealth/model_file.hpp"
#include "test_support.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <string_view>
#include <vector>

using namespace stealth::numeric;
namespace f = forecast_fixture;
using Bytes = std::vector<std::byte>;
namespace {
Bytes artifact() {
    auto s = std::as_bytes(std::span(f::artifact));
    return {s.begin(), s.end()};
}
void set32(Bytes &bytes, std::size_t at, std::uint32_t value) {
    for (unsigned int i = 0; i < 4; ++i)
        bytes[at + i] = static_cast<std::byte>((value >> (8 * i)) & 255U);
}
void rehash(Bytes &bytes) {
    const auto digest = stealth::detail::sha256(std::span(bytes).first(bytes.size() - 32));
    std::copy(digest.begin(), digest.end(), bytes.end() - 32);
}
void predictions(const ForecastModel &model) {
    ForecastWorkspace ws(model.config());
    std::array<float, f::tokens> out{};
    CHECK(model.forward(f::input, f::tokens, out, ws) == Error::ok);
    for (std::size_t i = 0; i < out.size(); ++i)
        test::near(out[i], f::expected[i], 5e-5, 2e-4);
}
void hash() {
    std::vector<std::byte> message(1048576);
    for (std::size_t i = 0; i < message.size(); ++i)
        message[i] = static_cast<std::byte>((i * 37 + 11) % 256);
    for (std::size_t j = 0; j < f::hash_lengths.size(); ++j) {
        const auto actual = stealth::detail::sha256(std::span(message).first(f::hash_lengths[j]));
        for (std::size_t i = 0; i < 32; ++i)
            CHECK(std::to_integer<unsigned int>(actual[i]) == f::hashes[j][i]);
    }
    constexpr std::array<unsigned char, 32> abc{0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
                                                0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
                                                0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
                                                0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad};
    const auto actual = stealth::detail::sha256(std::as_bytes(std::span("abc", 3)));
    for (std::size_t i = 0; i < 32; ++i)
        CHECK(std::to_integer<unsigned int>(actual[i]) == abc[i]);
}
void valid() {
    for (auto kernel : {DenseKernel::reference, DenseKernel::packed}) {
        auto model = load_forecaster(artifact(), kernel);
        predictions(*model);
        CHECK(model->config().features == 7);
        CHECK(model->config().horizon == 4);
    }
}
void corruption() {
    auto bytes = artifact();
    auto active = load_forecaster(bytes);
    auto *before = active.get();
    for (std::size_t n = 0; n < bytes.size(); ++n) {
        test::throws<std::invalid_argument>(
            [&] { active = load_forecaster(std::span(bytes).first(n)); });
        CHECK(active.get() == before);
    }
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        auto corrupt = bytes;
        corrupt[i] ^= std::byte{1};
        test::throws<std::invalid_argument>([&] { active = load_forecaster(corrupt); });
        CHECK(active.get() == before);
    }
    auto extra = bytes;
    extra.push_back(std::byte{0});
    test::throws<std::invalid_argument>([&] { active = load_forecaster(extra); });
    Bytes huge(max_model_file_bytes + 1);
    test::throws<std::length_error>([&] { active = load_forecaster(huge); });
    CHECK(active.get() == before);
    predictions(*active);
}
void semantics() {
    auto active = load_forecaster(artifact());
    auto *before = active.get();
    const std::array<std::pair<std::size_t, std::uint32_t>, 21> bad{{{0, 0},
                                                                     {8, 2},
                                                                     {12, 0x04030201U},
                                                                     {16, 60},
                                                                     {20, 0},
                                                                     {20, 65},
                                                                     {24, 513},
                                                                     {28, 3},
                                                                     {32, 2049},
                                                                     {36, 0},
                                                                     {36, 1025},
                                                                     {40, 0},
                                                                     {40, 1025},
                                                                     {44, 7},
                                                                     {48, 0},
                                                                     {48, 0x7fc00000U},
                                                                     {52, 0xffffffffU},
                                                                     {56, 1},
                                                                     {60, 1},
                                                                     {64 + 4 * 7, 0},
                                                                     {64 + 4 * 7, 0xbf800000U}}};
    for (auto [offset, value] : bad) {
        auto bytes = artifact();
        set32(bytes, offset, value);
        rehash(bytes);
        test::throws<std::logic_error>([&] { active = load_forecaster(bytes); });
        CHECK(active.get() == before);
    }
    for (std::size_t i = 0; i < f::parameters.size(); ++i) {
        auto bytes = artifact();
        set32(bytes, 64 + 4 * i, i % 2 == 0 ? 0x7fc00000U : 0x7f800000U);
        rehash(bytes);
        test::throws<std::invalid_argument>([&] { active = load_forecaster(bytes); });
        CHECK(active.get() == before);
    }
    predictions(*active);
}
void fuzz() {
    auto active = load_forecaster(artifact());
    test::Random rng;
    rng.state = 0x20261028U;
    for (int trial = 0; trial < 1500; ++trial) {
        auto bytes = artifact();
        (void)rng.next();
        const auto pos = static_cast<std::size_t>(rng.state) % bytes.size();
        (void)rng.next();
        bytes[pos] ^= static_cast<std::byte>((rng.state % 255U) + 1U);
        if (trial % 2 == 0)
            rehash(bytes); // Reach semantic parsing with a correct digest.
        try {
            auto candidate = load_forecaster(bytes);
            ForecastWorkspace ws(candidate->config());
            std::array<float, f::tokens> out{};
            out.fill(123);
            const auto status = candidate->forward(f::input, f::tokens, out, ws);
            if (status == Error::ok)
                for (float y : out)
                    CHECK(std::isfinite(y));
            else
                test::unchanged(out);
        } catch (const std::invalid_argument &) {
        } catch (const std::length_error &) {
        }
    }
    predictions(*active);
}
void file() {
    auto bytes = artifact();
    const auto directory = std::filesystem::path("model-file-test-artifacts");
    CHECK(std::filesystem::create_directory(directory));
    struct Cleanup {
        std::filesystem::path p;
        ~Cleanup() {
            std::error_code e;
            std::filesystem::remove_all(p, e);
        }
    } cleanup{directory};
    const auto path = directory / "model.stlf";
    {
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char *>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        CHECK(stream.good());
    }
    auto active = load_forecaster_file(path);
    predictions(*active);
    auto *before = active.get();
    {
        std::ofstream stream(path, std::ios::binary | std::ios::app);
        stream.put('x');
    }
    test::throws<std::logic_error>([&] { active = load_forecaster_file(path); });
    CHECK(active.get() == before);
    {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream.put('x');
    }
    test::throws<std::invalid_argument>([&] { active = load_forecaster_file(path); });
    CHECK(active.get() == before);
    {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream.seekp(static_cast<std::streamoff>(max_model_file_bytes));
        stream.put('x');
        CHECK(stream.good());
    }
    test::throws<std::length_error>([&] { active = load_forecaster_file(path); });
    CHECK(active.get() == before);
    std::filesystem::remove(path);
    test::throws<std::runtime_error>([&] { active = load_forecaster_file(path); });
    CHECK(active.get() == before);
    predictions(*active);
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("Expected test group");
        const std::string_view group = argv[1];
        if (group == "hash")
            hash();
        else if (group == "valid")
            valid();
        else if (group == "corruption")
            corruption();
        else if (group == "semantics")
            semantics();
        else if (group == "fuzz")
            fuzz();
        else if (group == "file")
            file();
        else
            throw std::runtime_error("Unknown group");
        std::cout << group << ": " << test::checks << " checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
