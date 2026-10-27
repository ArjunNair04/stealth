#include "stealth/runtime.hpp"

#include "stealth/build_config.hpp"

namespace stealth {

BuildInfo build_info() noexcept {
    return {detail::version, detail::compiler, detail::operating_system, detail::architecture,
            STEALTH_BUILD_CONFIGURATION};
}

} // namespace stealth
