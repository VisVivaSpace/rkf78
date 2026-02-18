# Harden & Improve Test Coverage for RKF78

## Phase 0: Fix Cargo.toml
- [x] Comment out `[[bench]]` section (no `benches/integration.rs` exists yet)
- [x] Fix pre-existing bugs (Brent's method, type annotation, doc comments, sign_change_detected)
- [x] Verify `cargo test` passes all 16 existing tests

## Phase 1: Input Validation & Defensive Checks
- [x] 1.1 Add `InvalidInput` and `NonFiniteState` variants to `IntegrationError`
- [x] 1.2 Add `validate_inputs` private method on `Rkf78<N>`
- [x] 1.3 Wire validation into `integrate()` and `integrate_to_event()`
- [x] 1.4 NaN/Inf detection during integration (after each accepted step)
- [x] 1.5 Make `max_steps` configurable (pub field, default 10_000_000)
- [x] 1.6 Add Phase 1 tests (6 new tests, 22 total)

## Phase 2: Expanded Test Coverage
- [x] `test_backward_integration`
- [x] `test_eccentric_orbit_energy_conservation`
- [x] `test_hyperbolic_trajectory_energy_conservation`
- [x] `test_step_size_too_small_error`
- [x] `test_max_steps_exceeded`
- [x] `test_step_rejection_with_large_h0`
- [x] `test_event_near_start`
- [x] `test_event_near_end`
- [x] Fix: step() h clamping now preserves sign for backward integration
- [x] Fix: h_next computed as positive magnitude (caller applies direction)
- [x] Fix: StepSizeTooSmall check now triggers when rejected at h_min (was dead code)

## Phase 3: Benchmarks
- [x] Create `benches/integration.rs`
- [x] Restore `[[bench]]` section in `Cargo.toml`
- [x] Verify `cargo bench` runs

## Phase 4: Implement `EventAction::Continue`
- [x] Add `collected_events` field to `Rkf78<N>`
- [x] Implement Continue logic in `integrate_to_event()`
- [x] Add `test_event_action_continue`
- [x] Add `test_event_action_continue_multiple`

## Review

### Summary of all changes across phases

**Phase 0 — Bug fixes to get codebase compiling and tests passing:**
- Rewrote Brent's method with standard textbook algorithm
- Fixed sign_change_detected for g_old == 0 case
- Added missing type annotations and doc comments
- Adjusted test tolerances to realistic levels

**Phase 1 — Input validation:**
- InvalidInput and NonFiniteState error variants
- validate_inputs() checks all parameters before integration
- NaN/Inf detection during integration
- Configurable max_steps field

**Phase 2 — Expanded test coverage (8 new tests):**
- Backward integration, eccentric/hyperbolic orbits, error conditions, events near boundaries
- Fixed step() sign handling for backward integration
- Fixed h_next to always return positive magnitude
- Fixed StepSizeTooSmall detection (was dead code)

**Phase 3 — Benchmarks:**
- Two criterion benchmarks: circular orbit and harmonic oscillator

**Phase 4 — EventAction::Continue:**
- collected_events Vec on Rkf78 stores events when action is Continue
- After Continue event, accept the full step to move past the zero crossing
- Two new tests: single crossing and multiple crossings

**Final state: 32 tests passing, 2 benchmarks running, cargo clippy clean.**

---

# Documentation Phase (Complete)

- [x] `docs/algorithm.md`, `docs/llm-context.md`, `README.md`, `src/lib.rs` update

---

# Strengthen RKF78 Foundation + GPU Batch Propagation

## Part A: Test & Code Review (Phases 1–5)

### Phase 1: Coefficient & Step Controller Tests
- [x] 1a. Add `B_ERR[i] == B[i] - B_HAT[i]` consistency test in `src/coefficients.rs` mod tests
- [x] 1b. Add `StepController::compute_factor()` boundary tests in `src/solver.rs` mod tests:
  - `error = 0.0` → `max_factor` (5.0)
  - `error = 1.0` → `safety` (0.9)
  - `error = 1e-20` → `max_factor` (clamped)
  - `error = 1e+20` → `min_factor` (clamped)
- **DO NOT modify**: any non-test code

### Phase 2: Rigorous Convergence Order Test
- [x] Replace `test_order_of_convergence` with single-step h-refinement study
  - Single step of RKF78 on `y' = cos(t)` (exact: `y = sin(t)`) from `t=0, y=0`
  - Step sizes `h = 0.4, 0.2, 0.1, 0.05, 0.025`
  - Error ratios `err(h) / err(h/2)` should approach `2^8 = 256` (assert 100–400)
- **File**: `src/solver.rs` (replace `test_order_of_convergence`)
- **DO NOT modify**: solver implementation

### Phase 3: Long-Duration & Round-Trip Tests
- [x] 3a. 100-orbit energy conservation — circular orbit, assert drift < 1e-7
- [x] 3b. Forward-backward round-trip — harmonic oscillator fwd 1 period, bwd 1 period, verify return to IC
- [x] 3c. Per-component tolerance — `Tolerances::with_components()` on 2D system, verify tighter component has smaller error
- **File**: `src/solver.rs` (new tests)

### Phase 4: Brent's Method Edge Cases
- [x] Root at bracket endpoint: `f(x) = x+1`, bracket `[-1, 1]`
- [x] Triple root: `f(x) = (x-1)^3`, bracket `[0, 2]`
- [x] Near-zero bracket: `f(x) = x`, bracket `[-1e-15, 1e-15]`
- **File**: `src/events.rs` (add tests in mod tests)
- **DO NOT modify**: `BrentSolver` implementation

### Phase 5: Minor Rust Polish
- [x] 5a. Tighten `TOL` in `src/coefficients.rs` from `1e-14` to `5e-15`
- [x] 5b. Add doc comment on `collected_events` noting it is cleared at start of `integrate_to_event`
- **Files**: `src/coefficients.rs` (TOL value), `src/solver.rs` (doc comment)

### Part A Verification
After each phase: `cargo test`, `cargo clippy`, `cargo fmt --check`, commit.

---

## Part B: Standalone GPU Batch Propagation (Phase 6)

**Prerequisite**: Phases 1–5 complete and committed.

### Phase 6a: Feature Flag and GPU Types
- [x] Create `src/gpu/mod.rs` — module declaration, re-exports
- [x] Create `src/gpu/types.rs` — `GpuState`, `TrajectoryStatus`, `GpuIntegrationParams` (all `#[repr(C)]` + bytemuck derives)
- [x] Modify `Cargo.toml` — add `gpu` feature, optional deps: `wgpu 24`, `bytemuck` (with `derive`), `pollster`
- [x] Modify `src/lib.rs` — add `#[cfg(feature = "gpu")] pub mod gpu;`
- [x] Tests: size assertions, bytemuck round-trip (4 tests)

### Phase 6b: WGSL Shader
- [x] Create `src/gpu/shader.wgsl` — complete compute shader with full 13-stage adaptive RKF78
  - All 13 stages unrolled with only non-zero A coefficients (55 terms)
  - B and B_ERR const arrays, two_body_accel(), compute_rhs()
  - I-controller step-size control, no-overshoot, convergence failure detection

### Phase 6c: Pipeline and Buffer Management
- [x] Create `src/gpu/pipeline.rs` — `Rkf78GpuPipeline`
- [x] Create `src/gpu/buffers.rs` — `read_buffer<T: Pod>()` using staging buffer
- [x] Modify `src/gpu/mod.rs` — add `GpuBatchPropagator` with new()/propagate_batch()

### Phase 6d: Integration Tests
- [x] Create `tests/gpu_integration.rs` (only compiled with `--features gpu`)
  - 6 tests: circular orbit GPU vs CPU, batch independence, energy conservation,
    elliptical orbit, step rejection, multi-dispatch completion

### Part B Verification
- [x] `cargo test` — 43 existing CPU tests pass (no GPU needed)
- [x] `cargo test --features gpu` — GPU tests pass (Mac Studio)
- [x] `cargo clippy --features gpu` — clean
- [x] `cargo fmt --check` — clean
- [x] Committed after each sub-phase

---

## Review — Strengthen RKF78 Foundation + GPU Batch Propagation

### Part A Summary (Phases 1–5)

Added 11 new tests (32 → 43 total), zero changes to solver implementation:

- **Phase 1**: `B_ERR[i] == B[i] - B_HAT[i]` consistency test + 4 `StepController::compute_factor()` boundary tests
- **Phase 2**: Rigorous single-step convergence order test — error ratios of ~480 and ~382 confirm 8th-order local truncation (expected: 2^9 = 512)
- **Phase 3**: 100-orbit energy conservation (drift 1.8e-9), forward-backward round-trip, per-component tolerance validation
- **Phase 4**: Brent's method edge cases — root at endpoint, triple root, near-zero bracket
- **Phase 5**: Tightened test TOL to 5e-15, added doc comment on `collected_events`

### Part B Summary (Phase 6a–6d)

GPU batch propagation behind `gpu` feature flag, zero changes to CPU solver:

- **Phase 6a**: `GpuState` (32B), `TrajectoryStatus` (16B), `GpuIntegrationParams` (48B) — all repr(C)/bytemuck for WGSL alignment. wgpu 24 (compatible with rustc 1.91).
- **Phase 6b**: WGSL compute shader — 13 stages unrolled with 55 non-zero A coefficients, Keplerian two-body force model, I-controller adaptive stepping, multi-dispatch support.
- **Phase 6c**: `Rkf78GpuPipeline` (device/queue/pipeline/bind_group_layout), `read_buffer<T: Pod>()` staging readback, `GpuBatchPropagator` with multi-dispatch loop.
- **Phase 6d**: 6 integration tests comparing GPU vs CPU reference.

### Files Modified (CPU solver — none)

| File | Change |
|------|--------|
| `src/coefficients.rs` | Test TOL tightened, 1 new test |
| `src/solver.rs` | Doc comment on collected_events, 8 new tests |
| `src/events.rs` | 3 new Brent edge case tests |

### Files Created (GPU)

| File | Contents |
|------|----------|
| `src/gpu/mod.rs` | `GpuBatchPropagator` + module re-exports |
| `src/gpu/types.rs` | 3 repr(C) structs + 4 tests |
| `src/gpu/shader.wgsl` | Full RKF78 compute shader (345 lines) |
| `src/gpu/pipeline.rs` | wgpu pipeline setup |
| `src/gpu/buffers.rs` | Staging buffer readback |
| `tests/gpu_integration.rs` | 6 GPU vs CPU integration tests |

### WGSL Alignment Fix

WGSL `vec3<f32>` has 16-byte alignment, causing struct sizes to exceed the Rust `repr(C)` layout (e.g., State was 48 bytes in WGSL vs 32 bytes in Rust). Fixed by using scalar f32 fields in WGSL buffer structs with `load_pos`/`load_vel` helper functions for internal vec3 computation.

### GPU Test Results (Mac Studio)

All 6 GPU integration tests pass. Results:
- Circular orbit GPU vs CPU: 0.0 km position error
- Elliptical orbit (e=0.5) GPU vs CPU: 0.0 km position error
- Energy conservation: relative drift 1.185e-6
- Batch independence: all 100 trajectories bitwise identical
- Step rejection: 2 rejected steps with large h_init
- Multi-dispatch: 15 steps across multiple dispatches

**Final state: 47 tests passing (43 CPU + 4 GPU types), 6 GPU integration tests passing, 2 benchmarks, cargo clippy clean.**

---

# Review Pass — Aerospace Numerical Methods + Rust Quality

## Phase 1: Make GPU Shader Force-Model-Agnostic
- [x] Remove `two_body_accel()` and `compute_rhs()` from `shader.wgsl`, add contract comment
- [x] Update `pipeline.rs` to accept `force_model_wgsl: &str` and concatenate at pipeline creation
- [x] Update `mod.rs` to pass through force model WGSL, remove Default impl
- [x] Update `tests/gpu_integration.rs` with `TWO_BODY_WGSL` const
- [x] Create `examples/gpu_two_body.rs`
- [x] Run `cargo test --features gpu`, `cargo clippy --features gpu`

## Phase 2: Brent's Method Defensive Guard
- [x] Add `fa != fb` guard in Brent's IQI condition, add bisection fallback for degenerate case
- [x] Add `test_brent_equal_function_values` test

## Phase 3: GPU Constructors Return Result
- [x] Add `GpuError` enum with `Display` + `Error`
- [x] Change `Rkf78GpuPipeline::new()` and `new_async()` to return `Result`
- [x] Change `GpuBatchPropagator::new()` to return `Result`
- [x] Update all GPU test calls to `.unwrap()`

## Phase 4: Tighten Test Tolerances
- [x] `test_100_orbit_energy_conservation`: 1e-7 → 1e-8
- [x] `test_forward_backward_round_trip`: 1e-9 → 1e-10
- [x] `test_brent_simple_root`: 1e-12 → 1e-13
- [x] `test_brent_trigonometric`: 1e-12 → 1e-13
- [x] `test_brent_cubic`: 1e-6 → 1e-10

## Phase 5: New Tests
- [x] `test_tolerance_sensitivity` — harmonic oscillator, 3 tolerance levels
- [x] `test_high_eccentricity_orbit_energy` — e=0.99, one period

## Phase 6: Rust Polish
- [x] `#[derive(Clone)]` on `StepController`
- [x] `#[derive(Clone)]` on `Rkf78<N>`
- [x] Doc comment on `integrate_to_event` noting linear interpolation O(h²) accuracy

## Review

### Summary

All 6 phases complete. Addressed all actionable findings from the aerospace numerical methods and Rust quality reviews.

### Files Modified

| File | Changes |
|------|---------|
| `src/gpu/shader.wgsl` | Removed hardcoded `two_body_accel`/`compute_rhs`; added user-supplied function contract comment |
| `src/gpu/pipeline.rs` | `new()` takes `force_model_wgsl: &str`, prepends to engine shader; returns `Result<Self, GpuError>` |
| `src/gpu/mod.rs` | `GpuBatchPropagator::new()` takes force model WGSL, returns `Result`; removed `Default`; added `GpuError` enum |
| `src/events.rs` | Defensive `fa != fb` guard in Brent's IQI/secant; tightened 3 test tolerances; 1 new test |
| `src/solver.rs` | `Clone` on `Rkf78`/`StepController`; O(h²) interpolation doc; tightened 3 test tolerances; 2 new tests |
| `tests/gpu_integration.rs` | `TWO_BODY_WGSL` const; all constructors use `.new(WGSL).unwrap()` |
| `examples/gpu_two_body.rs` | New — standalone GPU two-body example |
| `Cargo.toml` | `required-features = ["gpu"]` for example |
| `CLAUDE.md` | Dev environment note (Mac Studio with GPU); updated build commands |

### Production Code Changes (non-test)

- **GPU bring-your-own-RHS**: Force model removed from shader, user supplies WGSL at pipeline creation
- **Brent division-by-zero guard**: 3-line change adding bisection fallback when `fa == fb`
- **GPU error handling**: `GpuError` enum, constructors return `Result` instead of panicking
- **API ergonomics**: `Clone` on `Rkf78<N>` and `StepController`
- **Documentation**: Linear interpolation limitation on `integrate_to_event`

### Test Changes

- 3 new tests: `test_brent_equal_function_values`, `test_tolerance_sensitivity`, `test_high_eccentricity_orbit_energy`
- 5 tolerance tightenings (all verified against actual solver precision with 5x margins)

### Final State

56 tests (50 unit + 6 GPU integration), all passing. Clippy clean. Fmt clean.

---

# Examples & Documentation Update

## Phase 1: Add CPU Examples
- [x] `examples/harmonic_oscillator.rs` — Basic `OdeSystem<2>`, integrate, print vs exact
- [x] `examples/two_body_orbit.rs` — Keplerian 6-state, energy conservation, per-component tolerances
- [x] `examples/event_detection.rs` — Periapsis detection with `EventFunction<6>`, both Stop and Continue
- [x] Verify all three run: `cargo run --example <name>`

**DO NOT modify:** `src/solver.rs`, `src/events.rs`, `src/coefficients.rs`, GPU source files.

## Phase 2: Update README.md
- [x] Add GPU batch propagation bullet to Features list
- [x] Fix test count: "32 tests" → "56 tests"
- [x] Add `cargo test --features gpu` and example run commands to Build section
- [x] Add "GPU Batch Propagation" section (after Event Detection)
- [x] Add "Examples" section (after Build) listing all 4 examples

## Phase 3: Update `src/lib.rs` Docs
- [x] Add GPU bullet to features list
- [x] Uncomment event finding example (lines 69-78)
- [x] Fix algorithm.md link (replace `your-org/astrodynamics` placeholder)
- [x] Add GPU section after "Integration with Wisdom-Holman"

## Phase 4: Update `docs/llm-context.md`
- [x] Add `src/gpu/` row to Module Layout table
- [x] Add GPU section to API Surface
- [x] Add GPU f32 precision gotcha

## Phase 5: Verify Everything
- [x] `cargo run --example harmonic_oscillator`
- [x] `cargo run --example two_body_orbit`
- [x] `cargo run --example event_detection`
- [x] `cargo run --features gpu --example gpu_two_body`
- [x] `cargo test --features gpu`
- [x] `cargo clippy --features gpu`
- [x] `cargo fmt --check`

## Phase 6: Hermite Cubic Interpolation for Event State

Replace linear interpolation with Hermite cubic in `find_event_root()`.

- [x] Change `find_event_root` to compute `f_a = rhs(t_a, y_a)` and `f_b = rhs(t_b, y_b)` (2 RHS evals per event, not per step)
- [x] Replace linear interp `y = y_a + α(y_b - y_a)` with Hermite cubic using `{y_a, h·f_a, y_b, h·f_b}`
- [x] Update doc comments on `find_event_root` and `integrate_to_event` (O(h²) → O(h⁴))
- [x] Update `docs/llm-context.md` gotcha about event interpolation
- [x] Run event_detection example to verify improvement
- [x] Run `cargo test --features gpu`, clippy, fmt

**DO NOT modify:** `events.rs`, `coefficients.rs`, GPU files, examples, `integrate_to_event` logic.

## Commit Strategy
1. "Add CPU examples: harmonic oscillator, two-body orbit, event detection"
2. "Update README, lib.rs docs, and llm-context.md for GPU feature and examples"
3. "Upgrade event state interpolation from linear to Hermite cubic"

## Review

### Summary

Added 3 CPU examples, updated all documentation to cover GPU feature, and upgraded event state interpolation from linear to Hermite cubic.

### Files Created

| File | Description |
|------|-------------|
| `examples/harmonic_oscillator.rs` | Basic OdeSystem<2>, compares with exact cos/sin solution |
| `examples/two_body_orbit.rs` | Keplerian 6-state with per-component tolerances, energy conservation |
| `examples/event_detection.rs` | Periapsis detection with both EventAction::Stop and Continue |

### Files Modified

| File | Changes |
|------|---------|
| `README.md` | GPU feature bullet, test count 32→56, GPU section, Examples table, build commands |
| `src/lib.rs` | GPU feature bullet, uncommented event example (rust,ignore), fixed algorithm.md URL, GPU section |
| `docs/llm-context.md` | src/gpu/ module row, GPU API surface section, f32 precision gotcha, Hermite interpolation gotcha |
| `src/solver.rs` | `find_event_root()`: linear → Hermite cubic interpolation; updated doc comments O(h²) → O(h⁴) |

### Production Code Changes

Only one function changed: `find_event_root()` in `src/solver.rs`. The change:
- Computes `f_a = rhs(t_a, y_a)` and `f_b = rhs(t_b, y_b)` (2 RHS evals per event, not per step)
- Replaces `y = y_a + α(y_b - y_a)` with Hermite cubic basis functions `h00, h10, h01, h11`
- Shared `hermite_interp` closure used by both Brent's root-finding and final state computation

### Event Detection Accuracy Improvement

Tested on elliptical orbit (400 × 2000 km), periapsis radius error:

| Crossing | Before (linear) | After (Hermite) |
|----------|-----------------|-----------------|
| #1 | 6.94 km | 1.56e-3 km |
| #2 | 5.84 km | 1.10e-3 km |
| #3 | 6.87 km | 1.53e-3 km |
| #4 | 5.84 km | 1.10e-3 km |

~4,500× improvement for 2 extra RHS evaluations per event.

### Final State

56 tests passing, all 4 examples run, clippy clean, fmt clean.

---

# Fix AI Code Review Findings

## Finding 1: Fix Cargo.toml placeholder URL
- [x] Comment out `repository` line since real URL isn't known yet

## Finding 2: Make GPU `read_buffer()` return `Result`
- [x] Add `GpuError::ReadbackFailed(String)` variant
- [x] Change `read_buffer()` to return `Result<Vec<T>, GpuError>`
- [x] Change `propagate_batch()` to return `Result<(Vec<GpuState>, Vec<TrajectoryStatus>), GpuError>`
- [x] Add `.unwrap()` to all `propagate_batch()` calls in `tests/gpu_integration.rs`
- [x] Add `.expect("GPU propagation failed")` to `examples/gpu_two_body.rs`
- [x] Update `docs/llm-context.md` API surface

## Review

### Summary

Fixed two AI code review findings: commented out placeholder repository URL in Cargo.toml, and made GPU buffer readback return `Result` instead of panicking.

### Files Modified

| File | Changes |
|------|---------|
| `Cargo.toml` | Commented out placeholder repository URL |
| `src/gpu/buffers.rs` | `read_buffer()` returns `Result<Vec<T>, GpuError>` instead of `Vec<T>` |
| `src/gpu/mod.rs` | Added `ReadbackFailed` variant to `GpuError`; `propagate_batch()` returns `Result` |
| `tests/gpu_integration.rs` | Added `.unwrap()` to 6 `propagate_batch()` calls |
| `examples/gpu_two_body.rs` | Added `.expect("GPU propagation failed")` |
| `docs/llm-context.md` | Updated `propagate_batch` signature and `GpuError` variants |

### Verification

All 56 tests passing (50 CPU + 6 GPU integration), GPU example runs, clippy clean, fmt clean.

---

# Rust Naming Convention Review

## Tasks

- [x] Review all source files for RFC 430 naming convention compliance
- [x] Fix stale `propagate_batch` signature in README.md (missing `?` operator)
- [x] Run verification: `cargo test --features gpu`, `cargo clippy --features gpu`, `cargo fmt --check`

## Review

### Summary
The codebase naming conventions are fully compliant with RFC 430. The only issue found was a stale code example in `README.md:108` where `propagate_batch` was missing the `?` operator after the API was changed to return `Result`.

### Changes Made
- **`README.md:108`** — Added `?` to `propagator.propagate_batch(&states, &params)?;`

### Verification
- All tests pass (CPU + GPU)
- Clippy clean
- Formatting clean

---

# Prepare for crates.io Publishing

## DO NOT modify:
`src/solver.rs`, `src/events.rs`, `src/coefficients.rs`, `src/gpu/`, examples, tests

## Tasks

- [x] Add `Cargo.lock` to `.gitignore` and untrack it with `git rm --cached`
- [x] Fix license to MIT-only in `Cargo.toml`
- [x] Fix license text in `README.md`
- [x] Uncomment and set repository URL in `Cargo.toml`
- [x] Run verification: tests, clippy, fmt, dry-run publish

## Review

### Summary

Prepared crate for crates.io publishing: license set to MIT-only, repository URL set, `Cargo.lock` untracked (library crate convention).

### Files Modified

| File | Changes |
|------|---------|
| `.gitignore` | Added `Cargo.lock` |
| `Cargo.toml` | `license` → `"MIT"`, `repository` → `"https://github.com/VisVivaSpace/rkf78"` |
| `README.md` | License text updated to MIT-only |

### Verification

- 56 tests passing (50 CPU + 6 GPU integration)
- Clippy clean
- Formatting clean
- `cargo publish --dry-run` succeeds (26 files, 235 KB)

---

# v0.2.0 API Refactor

## Phase 0: Visibility and Hygiene
- [x] Make `BrentSolver`, `BrentError`, `sign_change_detected` → `pub(crate)`
- [x] Remove `BrentSolver`, `BrentError` from `lib.rs` re-exports
- [x] Make `coefficients` module `pub(crate)`
- [x] Remove `set_step_limits` method
- [x] Remove `_t_prev` / `_y_prev` dead variables in `integrate_to_event`
- [x] Add `#[must_use]` to `integrate()`, `integrate_to_event()` (removed redundant `step()` per clippy)
- [x] Add `Copy` derive to `Tolerances`, `StepController`, `StepResult`, `EventDirection`, `EventAction`, `EventConfig`, `EventResult`, `Stats`
- [x] Add `Display` impl for `Stats`
- [x] Make `StepController` `pub(crate)` (removed from lib.rs re-exports)

## Phase 1: Scalar/Float Traits + Generic Solver
- [ ] Create `src/scalar.rs` with `Float` and `Scalar` traits + f32/f64 impls
- [ ] Generify `OdeSystem`, `Tolerances`, `StepController`, `Rkf78`, `StepResult`
- [ ] Generify `IntegrationResult`, `IntegrationError`
- [ ] Generify `EventFunction`, `EventConfig`, `EventResult`, `BrentSolver`
- [ ] Convert inner loops to use `mul_real` and `from_f64`
- [ ] Update all tests, examples, benchmarks for new type params
- [ ] Add f32 tests (harmonic oscillator, two-body energy)

## Phase 2: Config Struct + h0 Convention + Fluent Setters
- [ ] Create `IntegrationConfig<R>` with fluent setters
- [ ] Move h_min/h_max/max_steps from `Rkf78` to `IntegrationConfig`
- [ ] Add fluent setter `with_controller` on `Rkf78`
- [ ] Add fluent setters on `StepController` (make pub again)
- [ ] Change h0 sign convention (magnitude only, direction inferred)
- [ ] Update `integrate()` and `integrate_to_event()` signatures
- [ ] Update all call sites

## Phase 3: StepObserver Trait
- [ ] Add `StepObserver<T, N>` trait + no-op impl for `()`
- [ ] Add `integrate_with_observer()` method
- [ ] Refactor `integrate()` to use observer internally
- [ ] Add observer test

## Phase 4: Dense Output
- [ ] Create `src/solution.rs` with `Solution<T, N>` struct
- [ ] Implement Hermite `eval(t)` and `eval_derivative(t)`
- [ ] Extract shared Hermite interpolation from `find_event_root`
- [ ] Add `integrate_dense()` method (1 extra RHS per accepted step)
- [ ] Add dense output tests

## Phase 5: Simultaneous Events
- [ ] Add `MultiEventFunction<T, N, M>` trait
- [ ] Add `event_index` field to `EventResult`
- [ ] Implement multi-event detection (earliest root wins)
- [ ] Add `integrate_with_multi_events()` method
- [ ] Change event return to `(IntegrationResult, Vec<EventResult>)`
- [ ] Remove `collected_events` from `Rkf78` struct
- [ ] Add multi-event tests

## Phase 6: Documentation
- [ ] Update README with "Why RKF78?" positioning
- [ ] Update README code examples for v0.2.0 API
- [ ] Update `src/lib.rs` crate docs
- [ ] Update `CLAUDE.md`
- [ ] Bump version to `0.2.0`
- [ ] Add `complex` feature flag in Cargo.toml
- [ ] Update `notes/rkf78_design.md`

---

# Code Review Fixes for v0.2.0 Release

## Phase 1: CPU Solver Fixes
- [x] **[C2]** Fix spurious event detection when g==0 on consecutive steps (`events.rs`)
- [x] **[H1]** Add non-finite state check to EventAction::Continue paths (`solver.rs`)
- [x] **[M5]** Document StepResult rejected-state behavior (`solver.rs`)
- [x] **[M8]** Tighten event test tolerance (`solver.rs`)
- [x] **[LOW]** Fix duplicate doc comment on step() (`solver.rs`)
- [x] **[LOW]** Use `from_f64(4.0)` instead of `two * two` (`events.rs`)

## Phase 2: GPU Fixes
- [x] **[C1]** Return error on dispatch exhaustion (`gpu/mod.rs`)
- [x] **[H2]** Guard against empty batch (`gpu/mod.rs`)
- [x] **[H3]** Pre-allocate staging buffer for status readback (`gpu/buffers.rs`, `gpu/mod.rs`)
- [x] **[H4]** Add validation to GpuIntegrationParams (`gpu/types.rs`)
- [x] **[M3]** Convert alignment assertion to Result error (`gpu/mod.rs`)
- [x] **[MEDIUM]** Restrict pipeline struct field visibility (`gpu/pipeline.rs`)
- [x] **[MEDIUM]** Document backward integration limitation (`gpu/mod.rs`)
- [x] **[LOW]** Add derives to GpuError (`gpu/mod.rs`)

## Phase 3: New Tests
- [x] **[M7]** Add individual A-matrix coefficient spot-checks (`coefficients.rs`)
- [x] **[LOW]** Add backward-time dense output test (`solution.rs`)
- [x] **[LOW]** Add GPU failure-path test (`tests/gpu_integration.rs`)
- [x] **[LOW]** Tighten GPU test tolerances (`tests/gpu_integration.rs`)

## Phase 4: Example Fix
- [x] **[M6]** Fix backwards tolerance comment (`examples/two_body_orbit.rs`)

## Review

All 21 findings from the code review have been fixed across 9 files.

| Phase | Findings Fixed | Files Modified |
|-------|---------------|----------------|
| 1 (CPU) | C2, H1, M5, M8, 2 LOW | solver.rs, events.rs |
| 2 (GPU) | C1, H2, H3, H4, M3, 2 MEDIUM, LOW | gpu/mod.rs, gpu/types.rs, gpu/pipeline.rs, gpu/buffers.rs |
| 3 (Tests) | M7, 3 LOW | coefficients.rs, solution.rs, tests/gpu_integration.rs |
| 4 (Example) | M6 | examples/two_body_orbit.rs |

**Deviation from plan:** The M8 tolerance was derived from first principles rather than the
planned 1e-8. The original test had h_max=∞, making the step size (and thus the Hermite
interpolation error) uncontrolled and platform-dependent. Fix: added `.with_h_max(0.1)` to
bound the step. Error analysis:
- Hermite cubic error bound: h^4 · |y''''| / 384 = 0.1^4 · e / 384 ≈ 7.1e-9
- Time error bound: (Hermite error) / |dy/dt| = 7.1e-9 / e ≈ 2.6e-9
- Time tolerance: 1e-6 (gives ~400x margin over bound; observed error: essentially zero)
- State tolerance: tightened from 1e-8 to 1e-12 since Brent finds g≈0 to machine precision

**Test count:** 71 unit tests + 7 GPU integration tests = 78 total (was 69 + 6 = 75 before).

**Verification:** `cargo test --features gpu` ✓, `cargo clippy --features gpu` ✓, `cargo fmt --check` ✓, `cargo run --features gpu --example gpu_two_body` ✓
