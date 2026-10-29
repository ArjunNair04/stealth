#pragma once

// These modes can optimize away isfinite checks even without any NaN input today.
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Stealth requires finite-value checks; disable fast-math and finite-math-only"
#endif
