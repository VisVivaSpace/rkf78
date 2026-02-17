//! # RKF78: Runge-Kutta-Fehlberg 7(8) Integrator
//!
//! A high-precision ODE integrator for smooth, non-stiff systems.
//! Widely used in orbital mechanics, control systems, oscillator analysis,
//! and anywhere high-order accuracy with adaptive stepping is needed.
//!
//! ## Features
//!
//! - 13-stage embedded RK7(8) pair providing 8th-order accuracy
//! - Adaptive step-size control with 7th-order error estimation
//! - Generic over `f32`/`f64` via [`Float`]/[`Scalar`] traits
//! - **Event finding** with Brent's method + Hermite cubic interpolation
//! - **Simultaneous multi-event** detection via [`MultiEventFunction`]
//! - **Dense output** via [`Solution`] with Hermite interpolation
//! - **Step observer** callback for trajectory inspection
//! - **GPU batch propagation** via `wgpu` compute shaders (optional `gpu` feature, `f32`)
//! - Zero runtime dependencies — based on NASA TR R-287 (Fehlberg, 1968)
//!
//! ## Basic Usage
//!
//! ```rust
//! use rkf78::{Rkf78, OdeSystem, Tolerances, IntegrationConfig};
//!
//! // Define your ODE system
//! struct HarmonicOscillator { omega: f64 }
//!
//! impl OdeSystem<f64, 2> for HarmonicOscillator {
//!     fn rhs(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
//!         dydt[0] = y[1];
//!         dydt[1] = -self.omega * self.omega * y[0];
//!     }
//! }
//!
//! // Set up and run the integrator
//! let sys = HarmonicOscillator { omega: 1.0 };
//! let tol = Tolerances::new(1e-12, 1e-12);
//! let mut solver = Rkf78::new(tol);
//!
//! let y0 = [1.0, 0.0];  // Initial conditions
//! let config = IntegrationConfig::new(0.0, 10.0, 0.1);
//! let (tf, yf) = solver.integrate(&sys, &config, &y0).unwrap();
//! ```
//!
//! ## Event Finding
//!
//! The integrator can detect when a user-defined event function crosses zero,
//! stopping precisely at that point. Common applications include:
//!
//! - Threshold crossings (state variable reaches a target value)
//! - Periodic event detection (zero-crossings of oscillating quantities)
//! - Boundary detection (system enters or exits a region)
//! - Terminal conditions (stop when a criterion is met)
//!
//! ```rust,ignore
//! use rkf78::{EventFunction, EventConfig, EventDirection, IntegrationResult, IntegrationConfig};
//!
//! struct ThresholdCrossing { value: f64 }
//!
//! impl EventFunction<f64, 2> for ThresholdCrossing {
//!     fn eval(&self, _t: f64, y: &[f64; 2]) -> f64 {
//!         y[0] - self.value
//!     }
//! }
//!
//! let event = ThresholdCrossing { value: 0.5 };
//! let config = EventConfig {
//!     direction: EventDirection::Falling,
//!     ..Default::default()
//! };
//!
//! let int_config = IntegrationConfig::new(t0, tf, h0);
//! let (result, _collected) = solver.integrate_to_event(&sys, &event, &config, &int_config, &y0)?;
//! match result {
//!     IntegrationResult::Event(ev) => println!("Event at t = {}", ev.t),
//!     IntegrationResult::Completed { t, .. } => println!("Reached tf = {}", t),
//! }
//! ```
//!
//! For simultaneous events, implement [`MultiEventFunction`] and use
//! [`Rkf78::integrate_with_multi_events()`].
//!
//! ## Dense Output
//!
//! Record the full trajectory and evaluate at arbitrary times:
//!
//! ```rust,ignore
//! let (tf, yf, solution) = solver.integrate_dense(&sys, &config, &y0)?;
//! let y_mid = solution.eval(5.0).unwrap();      // Hermite cubic interpolation
//! let dy_mid = solution.eval_derivative(5.0).unwrap();
//! ```
//!
//! ## Tolerance Selection
//!
//! - **High precision**: `atol = 1e-12`, `rtol = 1e-12` — reference solutions
//! - **Standard**: `atol = 1e-10`, `rtol = 1e-10` — general engineering
//! - **Fast**: `atol = 1e-6`, `rtol = 1e-6` — quick surveys and parameter sweeps
//!
//! For mixed-unit state vectors, use per-component tolerances via
//! [`Tolerances::with_components()`].
//!
//! ## GPU Batch Propagation
//!
//! With the `gpu` feature enabled, the crate provides [`gpu::GpuBatchPropagator`] for
//! propagating thousands of trajectories in parallel on the GPU via `wgpu` compute shaders.
//! The GPU solver uses `f32` precision (vs `f64` on CPU), making it suitable for Monte Carlo
//! studies, parameter sweeps, and trade studies where throughput matters more than
//! last-digit precision. Users supply their own WGSL force model at construction time.
//!
//! ## References
//!
//! 1. Fehlberg, E. (1968). "Classical Fifth-, Sixth-, Seventh-, and
//!    Eighth-Order Runge-Kutta Formulas with Stepsize Control".
//!    NASA TR R-287.
//!
//! 2. Hairer, E., Nørsett, S.P., & Wanner, G. (1993). "Solving
//!    Ordinary Differential Equations I: Nonstiff Problems".
//!    Springer.
//!
//! 3. Brent, R.P. (1973). "Algorithms for Minimization without
//!    Derivatives". Prentice-Hall.

#![deny(missing_docs)]
#![deny(unsafe_code)]

pub(crate) mod coefficients;
pub mod events;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod scalar;
pub mod solution;
pub mod solver;

pub use events::{
    EventAction, EventConfig, EventDirection, EventFunction, EventResult, MultiEventFunction,
};
pub use scalar::{Float, Scalar};
pub use solution::Solution;
pub use solver::{
    IntegrationConfig, IntegrationError, IntegrationResult, OdeSystem, Rkf78, Stats,
    StepController, StepObserver, StepResult, Tolerances,
};
