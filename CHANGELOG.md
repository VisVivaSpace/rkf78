# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-02-27

### Added

- Cross-validation tests comparing RKF78 against `ode_solvers` Dop853 (Dormand-Prince 8th-order)
  - Harmonic oscillator test (3.3e-11 agreement)
  - Two-body circular orbit test (0.66 μm agreement)
  - Two-body eccentric orbit test (2.0 km agreement after bug fix)
  - Energy conservation comparison test
- `ode_solvers` crate as dev-dependency for cross-validation

### Fixed

- Eccentric orbit cross-validation test: corrected `dx` parameter from 10.0 to 1.0
  - Improved agreement from 11.5 km to 2.0 km (5.7× better accuracy)
  - Root cause: eccentric orbits need finer time resolution at periapsis

## [0.2.0] - 2026-02-17

### Added

- **Generic scalar types** — `Float` and `Scalar` traits enable the same code to work with both `f32` and `f64`
- **`IntegrationConfig` struct** — bundles `t0`, `tf`, `h0`, `h_min`, `h_max`, `max_steps` with fluent setters; `h0` is always positive (direction inferred from `tf - t0`)
- **`StepObserver` trait** — callback after each accepted step for custom recording or analysis; no-op impl for `()` (zero overhead when unused)
- **Dense output** — `integrate_dense()` records the full trajectory; `Solution<T, N>` provides Hermite cubic interpolation at arbitrary times via `eval()` and `eval_derivative()`
- **Simultaneous multi-event detection** — `MultiEventFunction<T, N, M>` trait and `integrate_with_multi_events()` for monitoring M events concurrently, with earliest-wins semantics for `EventAction::Stop`
- **GPU batch propagation** — `GpuBatchPropagator` propagates thousands of trajectories in parallel via `wgpu` compute shaders (`f32`); user-supplied WGSL force model with generic force params at `@group(0) @binding(4)`
- **GPU parameter validation** — `GpuIntegrationParams::validate()` checks all fields are finite, positive, and consistent
- **GPU error variants** — `InvalidParams(String)` and `MaxDispatchesExhausted` for robust error handling
- Examples: `dense_output`, `step_observer`, `gpu_two_body`

### Changed

- Edition upgraded from 2021 to 2024 (MSRV 1.87)
- `integrate()` now takes `&IntegrationConfig` instead of individual `t0`/`tf`/`h0` arguments (**breaking**)
- `integrate_to_event()` signature updated for `IntegrationConfig` and returns `(IntegrationResult, Vec<EventResult>)` (**breaking**)
- Event state interpolation upgraded from linear to Hermite cubic (O(h⁴) accuracy)
- GPU force model parameters decoupled from `GpuIntegrationParams` into a user-defined generic `<P: Pod>` (**breaking**)
- GPU pipeline struct fields restricted to `pub(super)` / `pub(crate)` visibility
- GPU staging buffer pre-allocated and reused across dispatch loop iterations
- GPU alignment assertion converted to `Err(InvalidParams(...))` (no more panics)
- GPU dispatch loop fallthrough changed from `Ok(...)` to `Err(MaxDispatchesExhausted)`

### Fixed

- Spurious event detection when event function is zero on consecutive steps (`g_old == 0 && g_new == 0`)
- Non-finite state check missing in `EventAction::Continue` paths (both single and multi-event methods)
- Empty GPU batch now returns `Ok((vec![], vec![]))` instead of indexing out of bounds
- Event test tolerance derived from Hermite error analysis (`h_max` bounded, tolerance justified from `O(h⁴/384)`)
- `StepResult` docs now warn that `y` and `t` fields are populated even on rejected steps

## [0.1.0] - 2026-01-31

### Added

- Core RKF78 integrator with 13-stage embedded 7(8) pair from NASA TR R-287
- Adaptive step-size control with I-controller (safety 0.9, growth bounds [0.2×, 5.0×])
- Event detection via `EventFunction` trait with Brent's method root-finding
- `EventAction::Stop` and `EventAction::Continue` for terminal vs recording events
- `EventDirection::Rising`, `Falling`, `Any` for directional filtering
- Per-component tolerances via `Tolerances::with_components()`
- Backward integration support (negative time direction)
- GPU batch propagation via `wgpu` compute shaders (feature-gated: `gpu`)
- Comprehensive test suite (~50 tests) including energy conservation validation
- Algorithm documentation (`docs/algorithm.md`) with full mathematical derivation
- LLM context reference (`docs/llm-context.md`)
- Examples: `harmonic_oscillator`, `two_body_orbit`, `event_detection`
- Benchmarks via criterion

[0.2.1]: https://github.com/VisVivaSpace/rkf78/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/VisVivaSpace/rkf78/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/VisVivaSpace/rkf78/releases/tag/v0.1.0
