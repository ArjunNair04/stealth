# A successful ordinary compile is the control: reject only the unsafe mode.
set(probe "${STEALTH_BINARY_DIR}/fp_policy_probe.cpp")
set(object "${STEALTH_BINARY_DIR}/fp_policy_probe.o")
file(WRITE "${probe}" "#include \"${STEALTH_SOURCE_DIR}/src/fp_policy.hpp\"\nint probe() { return 0; }\n")
execute_process(COMMAND "${STEALTH_COMPILER}" -std=c++20 -c "${probe}" -o "${object}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Ordinary compile failed: ${output} ${error}")
endif()
foreach(flag -ffast-math -ffinite-math-only)
    execute_process(COMMAND "${STEALTH_COMPILER}" -std=c++20 "${flag}" -c "${probe}" -o "${object}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(result EQUAL 0 OR NOT "${error}" MATCHES "Stealth requires finite-value checks")
        message(FATAL_ERROR "Unsafe flag ${flag} was not rejected by the expected policy: ${output} ${error}")
    endif()
endforeach()
