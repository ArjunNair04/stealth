from pathlib import Path
p=Path('source/src/forecaster.cpp');s=p.read_text();s='#include "diagnostic_profile.hpp"\n'+s
# Single window: embedding, complete block, readout.
s=s.replace('    for (std::size_t t = 0; t < tokens; ++t) {','    ++stealth_diag::windows; auto stamp = stealth_diag::now();\n    for (std::size_t t = 0; t < tokens; ++t) {',1)
s=s.replace('    auto status = block_.forward(embedded, tokens, transformed, ws.block_, mode);','    stealth_diag::account(0,stamp);\n    auto status = block_.forward(embedded, tokens, transformed, ws.block_, mode);',1)
s=s.replace('    for (std::size_t t = first_row; t < tokens; ++t) {','    stealth_diag::account(1,stamp);\n    for (std::size_t t = first_row; t < tokens; ++t) {',1)
s=s.replace('    std::copy(candidate.begin() + static_cast<std::ptrdiff_t>(first_row), candidate.end(),','    stealth_diag::account(2,stamp);\n    std::copy(candidate.begin() + static_cast<std::ptrdiff_t>(first_row), candidate.end(),',1)
# Batch chunks: same three stages.
s=s.replace('    for (std::size_t b = 0; b < batch; b += ws.tile_) {','    stealth_diag::windows += batch;\n    for (std::size_t b = 0; b < batch; b += ws.tile_) {\n        auto stamp = stealth_diag::now();',1)
s=s.replace('        auto status = block_.forward_batch(', '        stealth_diag::account(0,stamp);\n        auto status = block_.forward_batch(',1)
s=s.replace('        for (std::size_t t = 0; t < windows * out_tokens; t += forecast_batch_rows) {','        stealth_diag::account(1,stamp);\n        for (std::size_t t = 0; t < windows * out_tokens; t += forecast_batch_rows) {',1)
s=s.replace('    }\n    std::copy(candidate.begin(), candidate.end(), output.begin());','        stealth_diag::account(2,stamp);\n    }\n    std::copy(candidate.begin(), candidate.end(), output.begin());',1)
p.write_text(s)
p=Path('source/src/transformer_block.cpp');s=p.read_text();s='#include "diagnostic_profile.hpp"\n'+s
s=s.replace('    for (std::size_t t = 0; t < tokens; t += tile) {','    auto stamp = stealth_diag::now();\n    for (std::size_t t = 0; t < tokens; t += tile) {',1)
s=s.replace('    auto status = causal_attention(query, key, value, tokens, context.subspan(first_row * width_),','    stealth_diag::account(3,stamp);\n    auto status = causal_attention(query, key, value, tokens, context.subspan(first_row * width_),',1)
s=s.replace('    for (std::size_t t = first_row; t < tokens; t += tile) {','    stealth_diag::account(4,stamp);\n    for (std::size_t t = first_row; t < tokens; t += tile) {',1)
s=s.replace('    std::copy(candidate.begin() + static_cast<std::ptrdiff_t>(first_row * width_), candidate.end(),','    stealth_diag::account(5,stamp);\n    std::copy(candidate.begin() + static_cast<std::ptrdiff_t>(first_row * width_), candidate.end(),',1)
s=s.replace('    for (std::size_t t = 0; t < batch * tokens; t += tile) {','    auto stamp = stealth_diag::now();\n    for (std::size_t t = 0; t < batch * tokens; t += tile) {',1)
s=s.replace('    // Reuse bounded attention scratch', '    stealth_diag::account(3,stamp);\n    // Reuse bounded attention scratch',1)
s=s.replace('    for (std::size_t t = 0; t < batch * out_tokens; t += tile) {','    stealth_diag::account(4,stamp);\n    for (std::size_t t = 0; t < batch * out_tokens; t += tile) {',1)
s=s.replace('    std::copy(candidate.begin(), candidate.end(), output.begin());','    stealth_diag::account(5,stamp);\n    std::copy(candidate.begin(), candidate.end(), output.begin());',1)
p.write_text(s)
