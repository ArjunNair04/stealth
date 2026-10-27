foreach(required STEALTH_BINARY_DIR STEALTH_SOURCE_DIR STEALTH_GENERATOR STEALTH_COMPILER STEALTH_BUILD_TYPE STEALTH_CTEST)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "Missing test input: ${required}")
    endif()
endforeach()
function(checked)
    execute_process(COMMAND ${ARGV} RESULT_VARIABLE result)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Package integration command failed: ${ARGV}")
    endif()
endfunction()
set(work "${STEALTH_BINARY_DIR}/install-smoke")
file(REMOVE_RECURSE "${work}")
file(MAKE_DIRECTORY "${work}")
checked("${CMAKE_COMMAND}" --install "${STEALTH_BINARY_DIR}" --config "${STEALTH_BUILD_TYPE}"
    --prefix "${work}/original")
# The original prefix disappears. This catches exported absolute install paths.
file(RENAME "${work}/original" "${work}/relocated package")
checked("${CMAKE_COMMAND}" -S "${STEALTH_SOURCE_DIR}/tests/consumer" -B "${work}/consumer"
    -G "${STEALTH_GENERATOR}" "-DCMAKE_CXX_COMPILER=${STEALTH_COMPILER}"
    "-DCMAKE_BUILD_TYPE=${STEALTH_BUILD_TYPE}" "-DCMAKE_PREFIX_PATH=${work}/relocated package")
checked("${CMAKE_COMMAND}" --build "${work}/consumer" --config "${STEALTH_BUILD_TYPE}")
checked("${STEALTH_CTEST}" --test-dir "${work}/consumer" -C "${STEALTH_BUILD_TYPE}" --output-on-failure)
