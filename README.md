# RKF78

A high-precision Runge-Kutta-Fehlberg 7(8) ODE integrator in Rust.

[![Crates.io](https://img.shields.io/crates/v/rkf78.svg)](https://crates.io/crates/rkf78)
[![docs.rs](https://docs.rs/rkf78/badge.svg)](https://docs.rs/rkf78)

## Why RKF78?

RKF78 is an 8th-order embedded Runge-Kutta method with 7th-order error estimation — one of the highest-order single-step integrators in common use. It takes large, accurate steps on smooth problems, making it ideal when you need high precision without a huge step count.

- **Zero runtime dependencies** — No nalgebra, no BLAS, no math libraries. Just Rust.
- **Const-generic `[T; N]` arrays** — Compile-time state dimension, zero heap allocation during integration.
- **Generic over f32/f64** — One codebase, two precision levels.
- **NASA-heritage coefficients** — From the original 1968 Fehlberg paper (NASA TR R-287).
- **Event detection** — Brent's method + Hermite cubic interpolation for precise zero-crossing location, simultaneous multi-event support, Stop/Continue actions.
- **Dense output** — Hermite cubic interpolation between steps for continuous trajectory evaluation.
- **Step observer** — Callback after each accepted step for custom recording or analysis.
- **GPU batch propagation** — Thousands of parallel integrations via `wgpu` compute shaders.

## When to use RKF78

RKF78 excels on **smooth, non-stiff** ODE systems where high accuracy is needed:

- Orbital mechanics and trajectory propagation
- Oscillators, wave equations, vibration analysis
- Celestial mechanics and N-body problems
- Chemical kinetics (non-stiff regimes)
- Control system simulation
- Any `dy/dt = f(t, y)` where you need better than 4th-order accuracy

If your problem is **stiff** (e.g., fast chemical reactions, circuit transients with widely-separated time scales), use an implicit method instead.

## Quick Start

```rust
use rkf78::{Rkf78, OdeSystem, Tolerances, IntegrationConfig};

// Define your ODE system: y'' + y = 0 (harmonic oscillator)
struct Oscillator { omega: f64 }

impl OdeSystem<f64, 2> for Oscillator {
    fn rhs(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
        dydt[0] = y[1];
        dydt[1] = -self.omega * self.omega * y[0];
    }
}

let sys = Oscillator { omega: 1.0 };
let tol = Tolerances::new(1e-12, 1e-12);
let mut solver = Rkf78::new(tol);

let y0 = [1.0, 0.0];  // y(0) = 1, y'(0) = 0
let config = IntegrationConfig::new(0.0, 10.0, 0.1);
let (tf, yf) = solver.integrate(&sys, &config, &y0).unwrap();
```

## Event Detection

Detect when a user-defined function crosses zero during integration.

```rust
use rkf78::{EventFunction, EventConfig, EventDirection, IntegrationResult, IntegrationConfig};

struct ThresholdCrossing { value: f64 }

impl EventFunction<f64, 2> for ThresholdCrossing {
    fn eval(&self, _t: f64, y: &[f64; 2]) -> f64 {
        y[0] - self.value
    }
}

let event = ThresholdCrossing { value: 0.5 };
let config = EventConfig {
    direction: EventDirection::Falling,
    ..Default::default()
};

let int_config = IntegrationConfig::new(0.0, 10.0, 0.1);
let (result, _collected) = solver.integrate_to_event(&sys, &event, &config, &int_config, &y0).unwrap();
match result {
    IntegrationResult::Event(ev) => println!("Event at t = {:.6}", ev.t),
    IntegrationResult::Completed { t, .. } => println!("No event, reached t = {}", t),
}
```

For simultaneous events, implement `MultiEventFunction<T, N, M>` and use `integrate_with_multi_events()`.

Events can also be configured with `EventAction::Continue` to record all crossings without stopping.

## Dense Output

Record the full trajectory and evaluate at arbitrary times:

```rust
let (tf, yf, solution) = solver.integrate_dense(&sys, &config, &y0).unwrap();

// Evaluate anywhere in [t0, tf] via Hermite cubic interpolation
let y_mid = solution.eval(5.0).unwrap();
let dy_mid = solution.eval_derivative(5.0).unwrap();
```

## Step Observer

Inspect each accepted step during integration:

```rust
use rkf78::StepObserver;

struct MaxErrorTracker(f64);

impl StepObserver<f64, 2> for MaxErrorTracker {
    fn on_step(&mut self, _t: f64, _y: &[f64; 2], _h: f64, error: f64) {
        self.0 = self.0.max(error);
    }
}

let mut tracker = MaxErrorTracker(0.0);
let (tf, yf) = solver.integrate_with_observer(&sys, &config, &y0, &mut tracker).unwrap();
```

## Tolerance Selection

| Precision Level | `atol` | `rtol` | Use Case |
|-----------------|--------|--------|----------|
| High | `1e-12` | `1e-12` | Reference solutions, precision analysis |
| Standard | `1e-10` | `1e-10` | General engineering analysis |
| Fast | `1e-6` | `1e-6` | Quick surveys, parameter sweeps |

For mixed-unit state vectors (e.g., position in km and velocity in km/s), use per-component tolerances via `Tolerances::with_components()`.

## Algorithm Details

For a full explanation of the mathematics — Butcher tableau, error estimation, step-size control, and Brent's method — see [`docs/algorithm.md`](docs/algorithm.md).

## Building and Testing

```bash
cargo build            # Build the crate
cargo test             # Run all tests (~78 tests)
cargo test --features gpu  # Include GPU tests (requires GPU)
cargo bench            # Run criterion benchmarks
cargo clippy           # Lint
cargo fmt --check      # Check formatting
cargo run --example harmonic_oscillator  # Run an example
```

## GPU Batch Propagation

With the `gpu` feature, RKF78 can propagate thousands of trajectories in parallel on the GPU using `wgpu` compute shaders. The GPU solver uses `f32` precision — suitable for Monte Carlo studies, parameter sweeps, and trade studies where throughput matters more than last-digit precision.

Users supply their own WGSL force model, which is prepended to the RKF78 compute shader at pipeline creation time.

```rust
use rkf78::gpu::{GpuBatchPropagator, GpuIntegrationParams, GpuState};

let propagator = GpuBatchPropagator::new(force_model_wgsl)?;
let (final_states, statuses) = propagator.propagate_batch(&states, &params, &force_params)?;
```

See [`examples/gpu_two_body.rs`](examples/gpu_two_body.rs) for a complete example with a Keplerian force model.

## Examples

| Example | Description |
|---------|-------------|
| [Harmonic Oscillator](examples/harmonic_oscillator.rs) | Basic integration with exact-solution comparison |
| [Dense Output](examples/dense_output.rs) | Continuous trajectory evaluation via Hermite interpolation |
| [Step Observer](examples/step_observer.rs) | Recording integration progress with a callback |
| [Two-Body Orbit](examples/two_body_orbit.rs) | Keplerian orbit with per-component tolerances |
| [Event Detection](examples/event_detection.rs) | Zero-crossing detection with Stop and Continue actions |
| [GPU Two-Body](examples/gpu_two_body.rs) | GPU batch propagation of multiple orbits |

See [`examples/README.md`](examples/README.md) for a suggested reading order.

## References

1. Fehlberg, E. (1968). *"Classical Fifth-, Sixth-, Seventh-, and Eighth-Order Runge-Kutta Formulas with Stepsize Control"*. NASA TR R-287.
2. Hairer, E., Norsett, S.P., & Wanner, G. (1993). *"Solving Ordinary Differential Equations I"*. Springer.
3. Brent, R.P. (1973). *"Algorithms for Minimization without Derivatives"*. Prentice-Hall.

## Development

This project was co-developed with [Claude](https://claude.ai), an AI assistant by Anthropic.

## License

Licensed under the [MIT License](LICENSE).
