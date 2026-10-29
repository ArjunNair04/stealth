#include "stealth/attention.hpp"
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
    using namespace stealth::numeric;
    // Equal query/key scores make this special case a running average.
    const std::array<float, 4> scores{}, values{10, 20, 30, 40}, changed{10, 20, 30, 400};
    const std::array<float, 4> expected{10, 15, 20, 25};
    std::array<float, 4> before{}, after{};
    AttentionWorkspace workspace(4, 1, 1);
    if (causal_attention(scores, scores, values, 4, before, workspace) != Error::ok ||
        causal_attention(scores, scores, changed, 4, after, workspace) != Error::ok)
        return 1;
    std::cout << "Attention with equal scores: each step averages only what it has seen.\n\n"
              << "Step  Input  Answer  Changed input  New answer\n"
              << std::fixed << std::setprecision(1);
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::cout << std::setw(4) << i + 1 << std::setw(7) << values[i] << std::setw(8) << before[i]
                  << std::setw(15) << changed[i] << std::setw(12) << after[i] << '\n';
        if (std::abs(before[i] - expected[i]) > 1e-5F || (i < 3 && before[i] != after[i]))
            return 1;
    }
    if (std::abs(after[3] - 115.0F) > 1e-5F)
        return 1;
    std::cout << "\nPASS: changing the future left all earlier answers unchanged.\n"
              << "This is a calculation example, not a trained market forecast.\n";
    return 0;
}
