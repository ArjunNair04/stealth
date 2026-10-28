#include <array>
#include <span>
#include <stealth/feed_forward.hpp>
#include <stealth/numeric.hpp>
#include <stealth/runtime.hpp>
#include <stealth/transformer_block.hpp>

int main() {
    using namespace stealth::numeric;
    const std::array<float, 1> w{2}, b{1}, x{3};
    std::array<float, 1> y{}, scratch{};
    Dense model(1, 1, w, b);
    if (model.forward(x, y, scratch) != Error::ok || y[0] != 7)
        return 1;
    const std::array<float, 1> zero{0}, one{1};
    FeedForward block(1, 1, {one, zero, zero, zero, zero, zero});
    FeedForwardWorkspace workspace(1, 1);
    if (block.forward(x, y, workspace) != Error::ok || y[0] != 3)
        return 1;
    AttentionWorkspace attention(1, 1, 1);
    if (causal_attention(zero, zero, x, 1, y, attention) != Error::ok || y[0] != 3)
        return 1;
    const std::array<float, 3> triple_zero{};
    TransformerBlock transformer(
        1, 1, 1,
        {one, zero, triple_zero, triple_zero, zero, zero, {one, zero, zero, zero, zero, zero}});
    TransformerWorkspace sequence(1, 1, 1, 1);
    if (transformer.forward(x, 1, y, sequence) != Error::ok || y[0] != 3)
        return 1;
    return stealth::build_info().version.empty() ? 1 : 0;
}
