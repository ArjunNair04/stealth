#pragma once

#include <string_view>

namespace stealth {

// Views refer to immutable strings owned by the library.
struct BuildInfo {
    std::string_view version;
    std::string_view compiler;
    std::string_view operating_system;
    std::string_view architecture;
    std::string_view configuration;
    std::string_view source_fingerprint;
};

[[nodiscard]] BuildInfo build_info() noexcept;

} // namespace stealth
