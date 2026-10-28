cmake_minimum_required(VERSION 3.25...4.0)
project(stealth VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

option(STEALTH_WARNINGS_AS_ERRORS "Treat project warnings as errors" OFF)
option(STEALTH_ENABLE_SANITIZERS "Enable address and undefined-behavior checks" OFF)
option(STEALTH_BUILD_BENCHMARKS "Build the optional native microbenchmark" OFF)
option(STEALTH_BUILD_EXAMPLES "Build the small runnable attention example" ON)
option(STEALTH_BUILD_CLI "Build the native command-line program" ON)

set(CMAKE_CXX_EXTENSIONS OFF)
if(STEALTH_ENABLE_SANITIZERS AND (MSVC OR NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU"))
    message(FATAL_ERROR "This sanitizer configuration requires GCC or Clang on macOS/Linux")
endif()

configure_file(cmake/build_config.hpp.in generated/stealth/build_config.hpp @ONLY)

function(stealth_development_options target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive- /Zc:__cplusplus)
        if(STEALTH_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
        if(STEALTH_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
    if(STEALTH_ENABLE_SANITIZERS)
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
endfunction()

add_library(stealth_core STATIC src/runtime.cpp src/numeric.cpp src/feed_forward.cpp src/attention.cpp src/transformer_block.cpp)
add_library(stealth::core ALIAS stealth_core)
set_target_properties(stealth_core PROPERTIES EXPORT_NAME core POSITION_INDEPENDENT_CODE ON)
target_compile_features(stealth_core PUBLIC cxx_std_20)
target_include_directories(stealth_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_compile_definitions(stealth_core PRIVATE STEALTH_BUILD_CONFIGURATION="$<CONFIG>")
stealth_development_options(stealth_core)
# Preserve explicit accumulation order; never trade finite checks for fast-math.
if(MSVC)
    target_compile_options(stealth_core PRIVATE /fp:precise)
else()
    target_compile_options(stealth_core PRIVATE -ffp-contract=off)
endif()

if(STEALTH_BUILD_CLI)
    add_executable(stealth_cli app/main.cpp)
    set_target_properties(stealth_cli PROPERTIES OUTPUT_NAME stealth)
    target_link_libraries(stealth_cli PRIVATE stealth::core)
    stealth_development_options(stealth_cli)
    install(TARGETS stealth_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

install(TARGETS stealth_core EXPORT stealthTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT stealthTargets FILE stealthTargets.cmake NAMESPACE stealth::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stealth)
configure_package_config_file(cmake/stealthConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/stealthConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stealth)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/stealthConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/stealthConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/stealthConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stealth)

if(STEALTH_BUILD_EXAMPLES)
    add_executable(stealth_attention_demo examples/attention_demo.cpp)
    target_link_libraries(stealth_attention_demo PRIVATE stealth::core)
    stealth_development_options(stealth_attention_demo)
    install(TARGETS stealth_attention_demo RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    if(BUILD_TESTING)
        add_test(NAME example.causal_average COMMAND stealth_attention_demo)
        set_tests_properties(example.causal_average PROPERTIES LABELS "integration;example")
    endif()
endif()

if(STEALTH_BUILD_BENCHMARKS)
    add_executable(stealth_bench bench/native_benchmark.cpp)
    target_link_libraries(stealth_bench PRIVATE stealth::core)
    stealth_development_options(stealth_bench)
endif()

if(BUILD_TESTING)
    add_executable(stealth_transformer_tests tests/transformer_tests.cpp)
    target_link_libraries(stealth_transformer_tests PRIVATE stealth::core)
    stealth_development_options(stealth_transformer_tests)
    foreach(group parity causality failures replay generated)
        add_test(NAME transformer.${group} COMMAND stealth_transformer_tests ${group})
        set_tests_properties(transformer.${group} PROPERTIES LABELS "integration;transformer")
    endforeach()
    add_executable(stealth_attention_tests tests/attention_tests.cpp)
    target_link_libraries(stealth_attention_tests PRIVATE stealth::core)
    stealth_development_options(stealth_attention_tests)
    foreach(group hand failures causality generated)
        add_test(NAME attention.${group} COMMAND stealth_attention_tests ${group})
        set_tests_properties(attention.${group} PROPERTIES LABELS "component;attention")
    endforeach()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU" AND NOT MSVC)
        add_test(NAME build.reject_fast_math COMMAND ${CMAKE_COMMAND}
            -DSTEALTH_COMPILER=${CMAKE_CXX_COMPILER}
            -DSTEALTH_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DSTEALTH_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/reject_fast_math.cmake)
        set_tests_properties(build.reject_fast_math PROPERTIES LABELS "component;build")
    endif()
    add_executable(stealth_allocation_tests tests/allocation_tests.cpp)
    target_link_libraries(stealth_allocation_tests PRIVATE stealth::core)
    stealth_development_options(stealth_allocation_tests)
    add_test(NAME memory.no_hot_allocations COMMAND stealth_allocation_tests)
    set_tests_properties(memory.no_hot_allocations PROPERTIES LABELS "integration;memory")
    add_executable(stealth_feed_forward_tests tests/feed_forward_tests.cpp)
    target_link_libraries(stealth_feed_forward_tests PRIVATE stealth::core)
    stealth_development_options(stealth_feed_forward_tests)
    foreach(group parity failures replay generated)
        add_test(NAME feed_forward.${group} COMMAND stealth_feed_forward_tests ${group})
        set_tests_properties(feed_forward.${group} PROPERTIES LABELS "integration")
    endforeach()
    add_executable(stealth_numeric_tests tests/numeric_tests.cpp)
    target_link_libraries(stealth_numeric_tests PRIVATE stealth::core)
    stealth_development_options(stealth_numeric_tests)
    foreach(group buffer dense dense_generated softmax softmax_generated normalization activation residual overlap)
        add_test(NAME numeric.${group} COMMAND stealth_numeric_tests ${group})
        set_tests_properties(numeric.${group} PROPERTIES LABELS "component")
    endforeach()
    add_executable(stealth_build_smoke tests/build_smoke.cpp)
    target_link_libraries(stealth_build_smoke PRIVATE stealth::core)
    stealth_development_options(stealth_build_smoke)
    add_test(NAME core.cxx20_link COMMAND stealth_build_smoke)
    if(STEALTH_BUILD_CLI)
        add_test(NAME cli.version COMMAND stealth_cli --version)
        set_tests_properties(cli.version PROPERTIES PASS_REGULAR_EXPRESSION "stealth ${PROJECT_VERSION}")
        add_test(NAME cli.build_info COMMAND stealth_cli --build-info)
        set_tests_properties(cli.build_info PROPERTIES PASS_REGULAR_EXPRESSION "cxx_standard: 20")
        add_test(NAME cli.invalid_argument COMMAND stealth_cli --unknown)
        set_tests_properties(cli.invalid_argument PROPERTIES WILL_FAIL TRUE)
    endif()
    # Sanitizers are a development build, not an installed distribution.
    if(NOT STEALTH_ENABLE_SANITIZERS)
        add_test(NAME package.relocated_consumer COMMAND ${CMAKE_COMMAND}
            -DSTEALTH_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -DSTEALTH_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DSTEALTH_GENERATOR=${CMAKE_GENERATOR}
            -DSTEALTH_COMPILER=${CMAKE_CXX_COMPILER}
            -DSTEALTH_BUILD_TYPE=$<CONFIG>
            -DSTEALTH_CTEST=${CMAKE_CTEST_COMMAND}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/install_consumer.cmake)
    endif()
endif()
