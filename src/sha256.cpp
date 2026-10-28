#include "sha256.hpp"
#include <algorithm>
#include <bit>
#include <cstdint>

namespace stealth::detail {
namespace {
constexpr std::array<std::uint32_t, 64> k{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
    0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
    0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
    0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
    0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
    0xc67178f2U};
void compress(std::span<const std::byte> block, std::array<std::uint32_t, 8> &state) noexcept {
    std::array<std::uint32_t, 64> w{};
    for (std::size_t i = 0; i < 16; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            w[i] = (w[i] << 8U) | std::to_integer<std::uint32_t>(block[4 * i + j]);
    for (std::size_t i = 16; i < 64; ++i) {
        const auto x = w[i - 15], y = w[i - 2];
        const auto s0 = std::rotr(x, 7) ^ std::rotr(x, 18) ^ (x >> 3U);
        const auto s1 = std::rotr(y, 17) ^ std::rotr(y, 19) ^ (y >> 10U);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    auto a = state[0], b = state[1], c = state[2], d = state[3], e = state[4], f = state[5],
         g = state[6], h = state[7];
    for (std::size_t i = 0; i < 64; ++i) {
        const auto s1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
        const auto choose = (e & f) ^ ((~e) & g);
        const auto t1 = h + s1 + choose + k[i] + w[i];
        const auto s0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
        const auto majority = (a & b) ^ (a & c) ^ (b & c);
        const auto t2 = s0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}
} // namespace
std::array<std::byte, 32> sha256(std::span<const std::byte> input) noexcept {
    std::array<std::uint32_t, 8> state{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                       0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    const auto size = input.size();
    while (input.size() >= 64) {
        compress(input.first(64), state);
        input = input.subspan(64);
    }
    std::array<std::byte, 128> tail{};
    std::copy(input.begin(), input.end(), tail.begin());
    tail[input.size()] = std::byte{0x80};
    const auto length = input.size() < 56 ? std::size_t{64} : std::size_t{128};
    const auto bits = static_cast<std::uint64_t>(size) * 8U;
    for (unsigned int i = 0; i < 8; ++i)
        tail[length - 1 - i] = static_cast<std::byte>((bits >> (8 * i)) & 255U);
    compress(std::span(tail).first(64), state);
    if (length == 128)
        compress(std::span(tail).subspan(64, 64), state);
    std::array<std::byte, 32> result{};
    for (std::size_t i = 0; i < 8; ++i)
        for (unsigned int j = 0; j < 4; ++j)
            result[4 * i + j] = static_cast<std::byte>((state[i] >> (24 - 8 * j)) & 255U);
    return result;
}
} // namespace stealth::detail
