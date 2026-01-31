# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RKF78 is a zero-dependency Runge-Kutta-Fehlberg 7(8) ODE integrator in Rust for spacecraft trajectory propagation. It provides an 8th-order solution with 7th-order error estimation using 13 stages per step, with adaptive step-size control and event detection via Brent's method. Coefficients come from NASA TR R-287 (Fehlberg, 1968).

Intended as both a standalone open-source crate and a sub-crate for a Wisdom-Holman symplectic integrator. See `notes/rkf78_design.md` for full design rationale and references.

## Development Environment

This project is developed on a Mac Studio with a GPU. Always run GPU tests — they are available on this machine.

## Build and Test Commands

```bash
cargo build                          # Build
cargo test                           # Run all CPU tests
cargo test --features gpu            # Run all tests including GPU
cargo test <test_name>               # Run a single test (e.g., cargo test test_two_body_energy_conservation)
cargo test -- --nocapture             # Run tests with stdout visible
cargo clippy --features gpu          # Lint (include GPU code)
cargo fmt --check                    # Check formatting
cargo bench                          # Run benchmarks (criterion)
```

## Architecture

Four source modules, all using const-generic `<const N: usize>` for compile-time state dimension:

- **`src/lib.rs`** — Public API re-exports and crate-level documentation
- **`src/coefficients.rs`** — Butcher tableau constants (`C`, `A`, `B`, `B_HAT`, `B_ERR`) from NASA TR R-287 Table X
- **`src/solver.rs`** — Core integrator: `Rkf78<N>`, `OdeSystem<N>` trait, `Tolerances<N>`, `StepController`, adaptive stepping, `integrate()` and `integrate_to_event()`
- **`src/events.rs`** — Event detection: `EventFunction<N>` trait, `BrentSolver`, `EventDirection`, `EventConfig`, sign-change monitoring with root-finding

The user implements `OdeSystem<N>` (provides `rhs(t, y, dydt)`) and optionally `EventFunction<N>` (provides `eval(t, y) -> f64`). The solver uses pre-allocated `[[f64; N]; 13]` workspace to avoid heap allocation during integration.

## Skills

These skills are available:
- **rust-mastery** — idiomatic Rust, ownership, borrowing, lifetimes, traits, generics
- **space-mission-design** — reference frames, coordinate systems, time systems, orbital mechanics
- **aerospace-numerical-methods** — IEEE 754 pitfalls, tolerance tiers, stable formulas, testing strategies for scientific computing

## Dependencies

**Runtime:** None. Zero runtime dependencies.

**Dev/test only:**
- **approx** — floating-point comparison macros for tests
- **criterion** — benchmarking framework

## Key Patterns

1. **Error handling**: `IntegrationError` enum (`StepSizeTooSmall`, `MaxStepsExceeded`, `EventFindingFailed`) with `Result<T, IntegrationError>` returns. `BrentError` for root-finding failures.
2. **Tolerance rule**: Non-iterative, deterministic computations (closed-form formulas, algebraic identities, pure arithmetic) must assert within floating-point precision (1e-10 to 1e-14). Only use loose tolerances for inherently approximate operations (iterative solvers, interpolated ephemeris data, coordinate conversions through trig functions) — and document why.
3. **Step control**: I-controller with safety factor 0.9, growth bounded [0.2×, 5.0×], exponent 1/8 for 8th-order method.

## Workflow Instructions

1. Think through the problem, read the codebase for relevant files, and write a plan to `tasks/todo.md`.
2. The plan should have a list of todo items that you can check off as you complete them.
3. For each step, the plan should list parts of the code that SHOULD NOT be modified while working on that step.
4. Before you begin working, check in with me and I will verify the plan. Allow me to chat about the plan before we start.
5. Then, begin working on the todo items, marking them as complete as you go.
6. Please every step of the way just give me a high level explanation of what changes you made.
7. Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
8. After you complete a major phase of the plan, add new tests as needed and commit the changes, then ask me to review your work. If user testing is needed, list what you want me to test.
9. Finally, add a review section to the `todo.md` file with a summary of the changes you made and any other relevant information.
10. DO NOT BE LAZY. NEVER BE LAZY. IF THERE IS A BUG FIND THE ROOT CAUSE AND FIX IT. NO TEMPORARY FIXES. YOU ARE A SENIOR DEVELOPER. NEVER BE LAZY
11. MAKE ALL FIXES AND CODE CHANGES AS SIMPLE AS HUMANLY POSSIBLE. THEY SHOULD ONLY IMPACT NECESSARY CODE RELEVANT TO THE TASK AND NOTHING ELSE. IT SHOULD IMPACT AS LITTLE CODE AS POSSIBLE. YOUR GOAL IS TO NOT INTRODUCE ANY BUGS. IT'S ALL ABOUT SIMPLICITY
