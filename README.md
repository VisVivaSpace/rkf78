# RKF78

A high-precision Runge-Kutta-Fehlberg 7(8) ODE integrator in Rust for spacecraft trajectory propagation.

<!-- Uncomment when published:
[![Crates.io](https://img.shields.io/crates/v/rkf78.svg)](https://crates.io/crates/rkf78)
[![docs.rs](https://docs.rs/rkf78/badge.svg)](https://docs.rs/rkf78)
-->

## Features

- **13-stage embedded RK7(8) pair** — 8th-order solution with 7th-order error estimation
- **Adaptive step-size control** — I-controller with safety factor, bounded growth
- **Event detection** — Sign-change monitoring with Brent's method root-finding
- **Zero runtime dependencies** — No external linear algebra or math libraries
- **Const-generic state dimension** — Compile-time optimization, no heap allocation during integration
- **NASA heritage coefficients** — From NASA TR R-287, Table X (Fehlberg, 1968)

## Quick Start

```rust
use rkf78::{Rkf78, OdeSystem, Tolerances};

// Define your ODE system: y'' + y = 0 (harmonic oscillator)
struct HarmonicOscillator { omega: f64 }

impl OdeSystem<2> for HarmonicOscillator {
    fn rhs(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
        dydt[0] = y[1];
        dydt[1] = -self.omega * self.omega * y[0];
    }
}

let sys = HarmonicOscillator { omega: 1.0 };
let tol = Tolerances::new(1e-12, 1e-12);
let mut solver = Rkf78::new(tol);

let y0 = [1.0, 0.0];  // y(0) = 1, y'(0) = 0
let (tf, yf) = solver.integrate(&sys, 0.0, &y0, 10.0, 0.1).unwrap();
```

## Event Detection

Detect when a user-defined function crosses zero during integration — essential for finding periapsis, eclipse boundaries, altitude crossings, etc.

```rust
use rkf78::{EventFunction, EventConfig, EventDirection, IntegrationResult};

struct ThresholdCrossing { value: f64 }

impl EventFunction<2> for ThresholdCrossing {
    fn eval(&self, _t: f64, y: &[f64; 2]) -> f64 {
        y[0] - self.value
    }
}

let event = ThresholdCrossing { value: 0.5 };
let config = EventConfig {
    direction: EventDirection::Falling,
    ..Default::default()
};

match solver.integrate_to_event(&sys, &event, &config, 0.0, &y0, 10.0, 0.1).unwrap() {
    IntegrationResult::Event(ev) => println!("Event at t = {:.6}", ev.t),
    IntegrationResult::Completed { t, .. } => println!("No event, reached t = {}", t),
}
```

Events can also be configured with `EventAction::Continue` to record all crossings without stopping.

## Tolerance Selection

| Precision Level | `atol` | `rtol` | Use Case |
|-----------------|--------|--------|----------|
| High | `1e-12` | `1e-12` | Orbit determination, precision propagation |
| Standard | `1e-10` | `1e-10` | General engineering analysis |
| Fast | `1e-6` | `1e-6` | Quick surveys, screening runs |

For mixed-unit state vectors (e.g., km and km/s), use per-component tolerances via `Tolerances::with_components()`.

**Validation**: At `tol = 1e-12`, energy drift for a Keplerian orbit is < 10⁻¹⁰ over one orbital period.

## Algorithm Details

For a full explanation of the mathematics — Butcher tableau, error estimation, step-size control, and Brent's method — see [`docs/algorithm.md`](docs/algorithm.md).

## Building and Testing

```bash
cargo build            # Build the crate
cargo test             # Run all tests (32 tests)
cargo bench            # Run criterion benchmarks
cargo clippy           # Lint
cargo fmt --check      # Check formatting
```

## References

1. Fehlberg, E. (1968). *"Classical Fifth-, Sixth-, Seventh-, and Eighth-Order Runge-Kutta Formulas with Stepsize Control"*. NASA TR R-287.
2. Hairer, E., Nørsett, S.P., & Wanner, G. (1993). *"Solving Ordinary Differential Equations I"*. Springer.
3. Brent, R.P. (1973). *"Algorithms for Minimization without Derivatives"*. Prentice-Hall.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
