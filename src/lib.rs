//! # RKF78: Runge-Kutta-Fehlberg 7(8) Integrator
//!
//! A high-precision ODE integrator for spacecraft trajectory propagation
//! and other astrodynamics applications.
//!
//! ## Features
//!
//! - 13-stage embedded RK7(8) pair providing 8th-order accuracy
//! - Adaptive step-size control with 7th-order error estimation
//! - **Event finding** with Brent's method for precise root location
//! - Based on NASA TR R-287 (Erwin Fehlberg, 1968)
//! - Minimal dependencies (no external linear algebra required)
//! - Designed for integration into larger astrodynamics libraries
//!
//! ## Basic Usage
//!
//! ```rust
//! use rkf78::{Rkf78, OdeSystem, Tolerances};
//!
//! // Define your ODE system
//! struct HarmonicOscillator { omega: f64 }
//!
//! impl OdeSystem<2> for HarmonicOscillator {
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
//! let (tf, yf) = solver.integrate(&sys, 0.0, &y0, 10.0, 0.1).unwrap();
//! ```
//!
//! ## Event Finding
//!
//! The integrator can detect when a user-defined event function crosses zero,
//! stopping precisely at that point. This is essential for astrodynamics
//! applications like detecting:
//!
//! - Periapsis/apoapsis (radial velocity = 0)
//! - Sphere of influence crossings
//! - Eclipse entry/exit
//! - Altitude threshold crossings
//!
//! ```rust
//! use rkf78::{Rkf78, OdeSystem, Tolerances, EventFunction, EventConfig, EventDirection, IntegrationResult};
//!
//! // Define an event (e.g., detect when y[0] crosses a threshold)
//! struct ThresholdCrossing { value: f64 }
//!
//! impl EventFunction<2> for ThresholdCrossing {
//!     fn eval(&self, _t: f64, y: &[f64; 2]) -> f64 {
//!         y[0] - self.value
//!     }
//! }
//!
//! // Configure event detection
//! let event = ThresholdCrossing { value: 0.5 };
//! let config = EventConfig {
//!     direction: EventDirection::Falling,  // Detect when y[0] decreases through 0.5
//!     ..Default::default()
//! };
//!
//! // // Integrate with event detection
//! // match solver.integrate_to_event(&sys, &event, &config, t0, &y0, tf, h0) {
//! //     Ok(IntegrationResult::Event(ev)) => {
//! //         println!("Event at t = {}, y = {:?}", ev.t, ev.y);
//! //     }
//! //     Ok(IntegrationResult::Completed { t, y }) => {
//! //         println!("Reached tf = {} without event", t);
//! //     }
//! //     Err(e) => eprintln!("Error: {}", e),
//! // }
//! ```
//!
//! ## Tolerance Selection
//!
//! Following NASA-STD-7009 guidance for numerical integration:
//!
//! - **Position (km)**: `atol ≈ 1e-12 km` for high-precision orbit determination
//! - **Velocity (km/s)**: `atol ≈ 1e-15 km/s` to match position precision
//! - **Relative tolerance**: Typically `1e-12` to `1e-14`
//!
//! For energy conservation tests with RKF78 at `tol=1e-12`:
//! - Energy drift should be `< 1e-10` over one orbital period
//!
//! ## Algorithm Details
//!
//! For a detailed explanation of the RKF7(8) mathematics — Butcher tableau,
//! error estimation, adaptive step-size control, and Brent's method for event
//! detection — see [`docs/algorithm.md`](https://github.com/your-org/astrodynamics/blob/master/docs/algorithm.md)
//! in the repository.
//!
//! ## Integration with Wisdom-Holman
//!
//! This integrator is designed to work as the perturbation integrator
//! in a Wisdom-Holman style mixed-variable symplectic scheme. In that
//! context, RKF78 handles the perturbation "kicks" while the Keplerian
//! motion is solved analytically.
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

pub mod coefficients;
pub mod events;
pub mod solver;

pub use events::{
    BrentError, BrentSolver, EventAction, EventConfig, EventDirection, EventFunction, EventResult,
};
pub use solver::{
    IntegrationError, IntegrationResult, OdeSystem, Rkf78, Stats, StepController, StepResult,
    Tolerances,
};
