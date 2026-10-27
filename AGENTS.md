# Stealth project instructions

This is a standalone private C++20 CPU forecasting project. Stealth is a working
name. Work in this checkout and keep generated outputs under ignored build/,
install/ or local/ directories. Do not use a temporary checkout as the main project.

Keep explanations simple and link design choices to their purpose. Prioritise
correctness, reproducibility and parsimony. Report measured results and limits.

Use portable CMake targets and C++20. Do not add a GPU, Python, platform SDK or
network dependency to the native core. Keep compiler paths and local preferences
in ignored CMakeUserPresets.json. Do not commit secrets, personal contact details,
local machine reports or raw market datasets. Use a GitHub noreply commit address.

Do not implement unrelated research while handling infrastructure. A Python
wrapper follows the native layers and calls the same core. Record pending work in
docs/tasks.md. Do not silently install new dependencies during a build.

For native changes, configure/build/test dev and release. Run sanitizers when
memory ownership or native execution changes. Changes to packaging require the
relocated consumer test. Run git diff --check before committing. Tests must fail
with a nonzero status, including Release builds; do not rely on disabled assert.

Preserve existing user changes. Keep this repository separate from QTS, and do not
modify QTS's canonical research records for infrastructure work here. No strategy,
performance or cross-platform support claim without corresponding evidence.
