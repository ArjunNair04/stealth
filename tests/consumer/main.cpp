#include <stealth/runtime.hpp>

#include <array>
#include <span>

int main() {
    const std::array versions{stealth::build_info().version};
    const std::span view(versions);
    return view.front().empty() ? 1 : 0;
}
