//! Runge-Kutta-Fehlberg 7(8) Integrator
//!
//! A 13-stage embedded RK7(8) pair for high-precision integration of ODEs.
//! Designed for spacecraft trajectory propagation and astrodynamics applications.
//!
//! Reference: NASA TR R-287, Erwin Fehlberg, 1968

use crate::coefficients::{A, B, B_ERR, C, STAGES};
use crate::events::{
    sign_change_detected, BrentError, BrentSolver, EventAction, EventConfig, EventFunction,
    EventResult,
};

/// System of ordinary differential equations: dy/dt = f(t, y)
pub trait OdeSystem<const N: usize> {
    /// Evaluate the right-hand side of the ODE system
    ///
    /// # Arguments
    /// * `t` - Current time
    /// * `y` - Current state vector
    /// * `dydt` - Output: derivative dy/dt
    fn rhs(&self, t: f64, y: &[f64; N], dydt: &mut [f64; N]);
}

/// Integration result from a single step
#[derive(Debug, Clone)]
pub struct StepResult<const N: usize> {
    /// New state after the step (8th order solution)
    pub y: [f64; N],
    /// New time value
    pub t: f64,
    /// Normalized error estimate (should be ≤ 1.0 for acceptance)
    pub error: f64,
    /// Suggested step size for next step
    pub h_next: f64,
    /// Whether the step was accepted
    pub accepted: bool,
}

/// Integration statistics for diagnostics
#[derive(Debug, Clone, Default)]
pub struct Stats {
    /// Total number of function evaluations
    pub fn_evals: u64,
    /// Number of accepted steps
    pub accepted_steps: u64,
    /// Number of rejected steps  
    pub rejected_steps: u64,
}

/// Step-size controller using an I-controller
///
/// h_new = safety * h * error^(-1/p)
/// where p = 8 for RKF78
#[derive(Clone)]
pub struct StepController {
    /// Safety factor (0.8-0.9 typical)
    pub safety: f64,
    /// Maximum growth factor per step
    pub max_factor: f64,
    /// Minimum reduction factor per step
    pub min_factor: f64,
    /// Exponent = 1/(order + 1) for I-controller
    exponent: f64,
}

impl Default for StepController {
    fn default() -> Self {
        Self {
            safety: 0.9,
            max_factor: 5.0,
            min_factor: 0.2,
            exponent: 1.0 / 8.0, // 1/(p+1) where p=7 for error estimate order
        }
    }
}

impl StepController {
    /// Compute the step size adjustment factor
    pub fn compute_factor(&self, error: f64) -> f64 {
        if error == 0.0 {
            return self.max_factor;
        }

        let factor = self.safety * error.powf(-self.exponent);
        factor.clamp(self.min_factor, self.max_factor)
    }
}

/// Tolerance specification for error control
///
/// Error is computed as: |y8 - y7| / (atol + rtol * |y8|)
#[derive(Debug, Clone)]
pub struct Tolerances<const N: usize> {
    /// Absolute tolerance per component
    pub atol: [f64; N],
    /// Relative tolerance per component
    pub rtol: [f64; N],
}

impl<const N: usize> Tolerances<N> {
    /// Create tolerances with uniform values
    pub fn new(atol: f64, rtol: f64) -> Self {
        Self {
            atol: [atol; N],
            rtol: [rtol; N],
        }
    }

    /// Create tolerances with per-component values
    pub fn with_components(atol: [f64; N], rtol: [f64; N]) -> Self {
        Self { atol, rtol }
    }
}

/// Runge-Kutta-Fehlberg 7(8) integrator
///
/// # Type Parameters
/// * `N` - Dimension of the state vector
///
/// # Example
/// ```ignore
/// use rkf78::{Rkf78, OdeSystem, Tolerances};
///
/// struct HarmonicOscillator { omega: f64 }
///
/// impl OdeSystem<2> for HarmonicOscillator {
///     fn rhs(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
///         dydt[0] = y[1];
///         dydt[1] = -self.omega * self.omega * y[0];
///     }
/// }
///
/// let tol = Tolerances::new(1e-12, 1e-12);
/// let mut solver = Rkf78::new(tol);
///
/// let sys = HarmonicOscillator { omega: 1.0 };
/// let y0 = [1.0, 0.0];
///
/// let (tf, yf) = solver.integrate(&sys, 0.0, &y0, 10.0, 0.1).unwrap();
/// ```
#[derive(Clone)]
pub struct Rkf78<const N: usize> {
    /// Tolerance specification
    tol: Tolerances<N>,
    /// Step-size controller
    controller: StepController,
    /// Minimum step size
    pub h_min: f64,
    /// Maximum step size
    pub h_max: f64,
    /// Maximum number of integration steps before error
    pub max_steps: u64,
    /// Stage evaluations (pre-allocated workspace)
    k: [[f64; N]; STAGES],
    /// Integration statistics
    pub stats: Stats,
    /// Events collected during `integrate_to_event` with `EventAction::Continue`.
    /// Cleared at the start of each `integrate_to_event` call.
    pub collected_events: Vec<EventResult<N>>,
}

impl<const N: usize> Rkf78<N> {
    /// Create a new RKF78 solver with specified tolerances
    pub fn new(tol: Tolerances<N>) -> Self {
        Self {
            tol,
            controller: StepController::default(),
            h_min: 1e-14,
            h_max: f64::INFINITY,
            max_steps: 10_000_000,
            k: [[0.0; N]; STAGES],
            stats: Stats::default(),
            collected_events: Vec::new(),
        }
    }

    /// Set minimum and maximum step sizes
    pub fn set_step_limits(&mut self, h_min: f64, h_max: f64) {
        self.h_min = h_min;
        self.h_max = h_max;
    }

    /// Perform a single integration step
    ///
    /// This computes the 13 stages, forms the 8th and 7th order solutions,
    /// estimates the error, and determines if the step should be accepted.
    pub fn step<S: OdeSystem<N>>(
        &mut self,
        sys: &S,
        t: f64,
        y: &[f64; N],
        h: f64,
    ) -> StepResult<N> {
        let h = h.signum() * h.abs().clamp(self.h_min, self.h_max);

        // Compute all 13 stages
        self.compute_stages(sys, t, y, h);

        // Compute 8th order solution
        let y8 = self.compute_solution(y, h);

        // Compute error estimate
        let error = self.compute_error(&y8, h);

        // Determine acceptance
        let accepted = error <= 1.0;

        // Compute next step size (always positive magnitude)
        let factor = self.controller.compute_factor(error);
        let h_next = (h.abs() * factor).clamp(self.h_min, self.h_max);

        // Update statistics
        self.stats.fn_evals += STAGES as u64;
        if accepted {
            self.stats.accepted_steps += 1;
        } else {
            self.stats.rejected_steps += 1;
        }

        StepResult {
            y: y8,
            t: t + h,
            error,
            h_next,
            accepted,
        }
    }

    /// Integrate from t0 to tf
    ///
    /// # Arguments
    /// * `sys` - The ODE system to integrate
    /// * `t0` - Initial time
    /// * `y0` - Initial state
    /// * `tf` - Final time
    /// * `h0` - Initial step size guess
    ///
    /// # Returns
    /// * `Ok((t_final, y_final))` on success
    /// * `Err(IntegrationError)` on failure
    pub fn integrate<S: OdeSystem<N>>(
        &mut self,
        sys: &S,
        t0: f64,
        y0: &[f64; N],
        tf: f64,
        h0: f64,
    ) -> Result<(f64, [f64; N]), IntegrationError> {
        if t0 == tf {
            return Ok((t0, *y0));
        }
        self.validate_inputs(t0, y0, tf, h0)?;

        let mut t = t0;
        let mut y = *y0;
        let mut h = h0;

        let direction = (tf - t0).signum();
        let mut step_count = 0u64;

        while (tf - t) * direction > self.h_min {
            // Don't overshoot the endpoint
            if (t + h - tf) * direction > 0.0 {
                h = tf - t;
            }

            let result = self.step(sys, t, &y, h);

            if result.accepted {
                t = result.t;
                y = result.y;
                if !y.iter().all(|v| v.is_finite()) {
                    return Err(IntegrationError::NonFiniteState { t });
                }
            }

            h = result.h_next * direction;

            step_count += 1;
            if step_count > self.max_steps {
                return Err(IntegrationError::MaxStepsExceeded);
            }

            // Check for step size too small: if the step was rejected and
            // the next step size is already at h_min, we can't make progress
            if !result.accepted && result.h_next <= self.h_min && (tf - t) * direction > self.h_min
            {
                return Err(IntegrationError::StepSizeTooSmall {
                    t,
                    h: result.h_next,
                });
            }
        }

        Ok((t, y))
    }

    /// Compute all 13 stages
    #[allow(clippy::needless_range_loop)]
    fn compute_stages<S: OdeSystem<N>>(&mut self, sys: &S, t: f64, y: &[f64; N], h: f64) {
        let mut y_temp = [0.0; N];

        // Stage 0: k[0] = f(t, y)
        sys.rhs(t, y, &mut self.k[0]);

        // Stages 1-12
        for i in 1..STAGES {
            // y_temp = y + h * sum_{j=0}^{i-1} a[i][j] * k[j]
            for n in 0..N {
                let mut sum = 0.0;
                for j in 0..i {
                    sum += A[i][j] * self.k[j][n];
                }
                y_temp[n] = y[n] + h * sum;
            }

            // k[i] = f(t + c[i]*h, y_temp)
            sys.rhs(t + C[i] * h, &y_temp, &mut self.k[i]);
        }
    }

    /// Compute the 8th order solution from the stages
    #[allow(clippy::needless_range_loop)]
    fn compute_solution(&self, y: &[f64; N], h: f64) -> [f64; N] {
        let mut y_new = [0.0; N];

        for n in 0..N {
            let mut sum = 0.0;
            for i in 0..STAGES {
                sum += B[i] * self.k[i][n];
            }
            y_new[n] = y[n] + h * sum;
        }

        y_new
    }

    /// Compute the normalized error estimate
    ///
    /// Uses the infinity norm of the scaled error:
    /// error = max_i( |h * sum_j (b[j] - b_hat[j]) * k[j][i]| / scale[i] )
    /// where scale[i] = atol[i] + rtol[i] * |y8[i]|
    #[allow(clippy::needless_range_loop)]
    fn compute_error(&self, y8: &[f64; N], h: f64) -> f64 {
        let mut max_err: f64 = 0.0;

        for n in 0..N {
            // Compute error in component n
            let mut err_n = 0.0;
            for i in 0..STAGES {
                err_n += B_ERR[i] * self.k[i][n];
            }
            err_n *= h;

            // Scale by tolerance
            let scale = self.tol.atol[n] + self.tol.rtol[n] * y8[n].abs();
            let scaled_err = err_n.abs() / scale;

            max_err = max_err.max(scaled_err);
        }

        max_err
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = Stats::default();
    }

    /// Validate integration inputs
    fn validate_inputs(
        &self,
        t0: f64,
        y0: &[f64; N],
        tf: f64,
        h0: f64,
    ) -> Result<(), IntegrationError> {
        if !t0.is_finite() || !tf.is_finite() || !h0.is_finite() {
            return Err(IntegrationError::InvalidInput {
                message: "t0, tf, and h0 must be finite".to_string(),
            });
        }
        if h0 == 0.0 {
            return Err(IntegrationError::InvalidInput {
                message: "h0 must be non-zero".to_string(),
            });
        }
        let direction = tf - t0;
        if direction != 0.0 && h0.signum() != direction.signum() {
            return Err(IntegrationError::InvalidInput {
                message: "h0 sign must match integration direction (tf - t0)".to_string(),
            });
        }
        for (i, &val) in y0.iter().enumerate() {
            if !val.is_finite() {
                return Err(IntegrationError::InvalidInput {
                    message: format!("y0[{}] is not finite", i),
                });
            }
        }
        for (i, (&a, &r)) in self.tol.atol.iter().zip(self.tol.rtol.iter()).enumerate() {
            if !a.is_finite() || a <= 0.0 {
                return Err(IntegrationError::InvalidInput {
                    message: format!("atol[{}] must be positive and finite", i),
                });
            }
            if !r.is_finite() || r < 0.0 {
                return Err(IntegrationError::InvalidInput {
                    message: format!("rtol[{}] must be non-negative and finite", i),
                });
            }
        }
        Ok(())
    }

    /// Integrate until an event occurs or the final time is reached.
    ///
    /// This method monitors an event function `g(t, y)` during integration.
    /// When `g` changes sign (crosses zero), Brent's method is used to
    /// precisely locate the time of the event.
    ///
    /// **Note:** The event state is found via Hermite cubic interpolation
    /// between integration steps, giving O(h⁴) accuracy in the event state.
    /// The event *time* is located to `root_tol` precision by Brent's method.
    ///
    /// # Arguments
    /// * `sys` - The ODE system to integrate
    /// * `event` - The event function to monitor
    /// * `config` - Configuration for event detection
    /// * `t0` - Initial time
    /// * `y0` - Initial state
    /// * `tf` - Final time (integration stops here if no event)
    /// * `h0` - Initial step size guess
    ///
    /// # Returns
    /// * `Ok(IntegrationResult::Event(event_result))` - Event was detected
    /// * `Ok(IntegrationResult::Completed(t, y))` - Reached tf without event
    /// * `Err(IntegrationError)` - Integration failed
    ///
    /// # Example
    ///
    /// ```ignore
    /// use rkf78::{Rkf78, OdeSystem, Tolerances, EventFunction, EventConfig};
    ///
    /// // Detect periapsis (radial velocity = 0, approaching)
    /// struct PeriapsisEvent;
    /// impl EventFunction<6> for PeriapsisEvent {
    ///     fn eval(&self, _t: f64, y: &[f64; 6]) -> f64 {
    ///         // Radial velocity: r_dot = (r · v) / |r|
    ///         let r = (y[0]*y[0] + y[1]*y[1] + y[2]*y[2]).sqrt();
    ///         (y[0]*y[3] + y[1]*y[4] + y[2]*y[5]) / r
    ///     }
    /// }
    ///
    /// let sys = TwoBodyProblem { mu: 398600.4418 };
    /// let event = PeriapsisEvent;
    /// let config = EventConfig {
    ///     direction: EventDirection::Rising,  // r_dot going from - to +
    ///     ..Default::default()
    /// };
    ///
    /// let result = solver.integrate_to_event(&sys, &event, &config, t0, &y0, tf, h0);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn integrate_to_event<S, E>(
        &mut self,
        sys: &S,
        event: &E,
        config: &EventConfig,
        t0: f64,
        y0: &[f64; N],
        tf: f64,
        h0: f64,
    ) -> Result<IntegrationResult<N>, IntegrationError>
    where
        S: OdeSystem<N>,
        E: EventFunction<N>,
    {
        if t0 == tf {
            return Ok(IntegrationResult::Completed { t: t0, y: *y0 });
        }
        self.validate_inputs(t0, y0, tf, h0)?;
        self.collected_events.clear();

        let mut t = t0;
        let mut y = *y0;
        let mut h = h0;

        let direction = (tf - t0).signum();

        // Evaluate initial event function
        let mut g_prev = event.eval(t, &y);

        // Store previous accepted state for interpolation
        let mut _t_prev = t;
        let mut _y_prev = y;

        let mut step_count = 0u64;

        while (tf - t) * direction > self.h_min {
            // Don't overshoot the endpoint
            if (t + h - tf) * direction > 0.0 {
                h = tf - t;
            }

            let result = self.step(sys, t, &y, h);

            if result.accepted {
                // Evaluate event function at new state
                let g_new = event.eval(result.t, &result.y);

                // Check for sign change
                if sign_change_detected(g_prev, g_new, config.direction) {
                    // Event detected! Use Brent's method to find precise time
                    let event_result = self.find_event_root(
                        sys, event, t, &y, result.t, &result.y, g_prev, g_new, config,
                    )?;

                    match config.action {
                        EventAction::Stop => {
                            return Ok(IntegrationResult::Event(event_result));
                        }
                        EventAction::Continue => {
                            // Record event, accept the full step (not the event point)
                            // so we move past the zero crossing and don't re-detect it
                            self.collected_events.push(event_result);
                            _t_prev = t;
                            _y_prev = y;
                            t = result.t;
                            y = result.y;
                            g_prev = g_new;
                            h = result.h_next * direction;
                            continue;
                        }
                    }
                }

                // Update state
                _t_prev = t;
                _y_prev = y;
                t = result.t;
                y = result.y;
                if !y.iter().all(|v| v.is_finite()) {
                    return Err(IntegrationError::NonFiniteState { t });
                }
                g_prev = g_new;
            }

            h = result.h_next * direction;

            step_count += 1;
            if step_count > self.max_steps {
                return Err(IntegrationError::MaxStepsExceeded);
            }

            if !result.accepted && result.h_next <= self.h_min && (tf - t) * direction > self.h_min
            {
                return Err(IntegrationError::StepSizeTooSmall {
                    t,
                    h: result.h_next,
                });
            }
        }

        // Reached tf without event
        Ok(IntegrationResult::Completed { t, y })
    }

    /// Find the precise root location using Brent's method.
    ///
    /// The event state is interpolated via Hermite cubic using the RHS
    /// evaluations at the step endpoints, giving O(h⁴) state accuracy.
    #[allow(clippy::too_many_arguments)]
    fn find_event_root<S, E>(
        &mut self,
        sys: &S,
        event: &E,
        t_a: f64,
        y_a: &[f64; N],
        t_b: f64,
        y_b: &[f64; N],
        g_a: f64,
        g_b: f64,
        config: &EventConfig,
    ) -> Result<EventResult<N>, IntegrationError>
    where
        S: OdeSystem<N>,
        E: EventFunction<N>,
    {
        let solver = BrentSolver::new(config.root_tol, config.max_iter);

        // Compute RHS at both endpoints for Hermite cubic interpolation.
        // Cost: 2 RHS evaluations per event (not per step).
        let mut f_a = [0.0; N];
        let mut f_b = [0.0; N];
        sys.rhs(t_a, y_a, &mut f_a);
        sys.rhs(t_b, y_b, &mut f_b);
        self.stats.fn_evals += 2;

        let dt = t_b - t_a;

        // Hermite cubic interpolation: given y_a, f_a, y_b, f_b,
        // compute y(t) for t in [t_a, t_b] with O(h⁴) accuracy.
        let hermite_interp = |t: f64| -> [f64; N] {
            let alpha = (t - t_a) / dt;
            let a2 = alpha * alpha;
            let a3 = a2 * alpha;
            // Hermite basis functions
            let h00 = 1.0 - 3.0 * a2 + 2.0 * a3; // y_a weight
            let h10 = alpha - 2.0 * a2 + a3; // f_a weight (scaled by dt)
            let h01 = 3.0 * a2 - 2.0 * a3; // y_b weight
            let h11 = -a2 + a3; // f_b weight (scaled by dt)

            let mut y = [0.0; N];
            for i in 0..N {
                y[i] = h00 * y_a[i] + h10 * dt * f_a[i] + h01 * y_b[i] + h11 * dt * f_b[i];
            }
            y
        };

        // Create a function that evaluates g at time t using Hermite interpolation
        let eval_g = |t: f64| {
            let y_interp = hermite_interp(t);
            event.eval(t, &y_interp)
        };

        // Find the root
        match solver.find_root(eval_g, t_a, t_b, Some(g_a), Some(g_b)) {
            Ok((t_event, g_value, iterations)) => {
                let y_event = hermite_interp(t_event);
                Ok(EventResult {
                    t: t_event,
                    y: y_event,
                    g_value,
                    iterations,
                })
            }
            Err(BrentError::NotBracketed { .. }) => {
                // This shouldn't happen since we already detected a sign change
                Err(IntegrationError::EventFindingFailed {
                    message: "Root not bracketed despite sign change detection".to_string(),
                })
            }
            Err(BrentError::MaxIterations {
                current_best,
                f_value,
                iterations,
            }) => {
                // Return best estimate even if not fully converged
                let y_event = hermite_interp(current_best);
                Ok(EventResult {
                    t: current_best,
                    y: y_event,
                    g_value: f_value,
                    iterations,
                })
            }
        }
    }
}

/// Result of integration with event detection
#[derive(Debug, Clone)]
pub enum IntegrationResult<const N: usize> {
    /// Integration completed normally (reached final time)
    Completed {
        /// Final time
        t: f64,
        /// Final state vector
        y: [f64; N],
    },
    /// Integration stopped at an event
    Event(EventResult<N>),
}

/// Errors that can occur during integration
#[derive(Debug, Clone)]
pub enum IntegrationError {
    /// Step size became too small
    StepSizeTooSmall {
        /// Time at which step size became too small
        t: f64,
        /// Step size that was too small
        h: f64,
    },
    /// Maximum number of steps exceeded
    MaxStepsExceeded,
    /// Event finding failed
    EventFindingFailed {
        /// Description of the failure
        message: String,
    },
    /// Invalid input parameters
    InvalidInput {
        /// Description of the invalid input
        message: String,
    },
    /// Non-finite state detected during integration
    NonFiniteState {
        /// Time at which non-finite state was detected
        t: f64,
    },
}

impl std::fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntegrationError::StepSizeTooSmall { t, h } => {
                write!(f, "Step size {} too small at t = {}", h, t)
            }
            IntegrationError::MaxStepsExceeded => {
                write!(f, "Maximum number of integration steps exceeded")
            }
            IntegrationError::EventFindingFailed { message } => {
                write!(f, "Event finding failed: {}", message)
            }
            IntegrationError::InvalidInput { message } => {
                write!(f, "Invalid input: {}", message)
            }
            IntegrationError::NonFiniteState { t } => {
                write!(f, "Non-finite state detected at t = {}", t)
            }
        }
    }
}

impl std::error::Error for IntegrationError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Harmonic oscillator: y'' + ω²y = 0
    /// State: [y, y']
    struct HarmonicOscillator {
        omega: f64,
    }

    impl OdeSystem<2> for HarmonicOscillator {
        fn rhs(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
            dydt[0] = y[1];
            dydt[1] = -self.omega * self.omega * y[0];
        }
    }

    #[test]
    fn test_harmonic_oscillator() {
        let omega = 1.0;
        let sys = HarmonicOscillator { omega };

        // Initial conditions: y(0) = 1, y'(0) = 0
        // Exact solution: y = cos(ωt), y' = -ω*sin(ωt)
        let y0 = [1.0, 0.0];
        let t0 = 0.0;
        let tf = 2.0 * std::f64::consts::PI; // One period

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);

        let (t_final, y_final) = solver.integrate(&sys, t0, &y0, tf, 0.1).unwrap();

        // Should return to initial conditions after one period
        assert!((t_final - tf).abs() < 1e-10);
        assert!(
            (y_final[0] - 1.0).abs() < 1e-10,
            "y(2π) = {}, expected 1.0",
            y_final[0]
        );
        assert!(
            y_final[1].abs() < 1e-10,
            "y'(2π) = {}, expected 0.0",
            y_final[1]
        );

        println!("Harmonic oscillator test passed:");
        println!("  Final y = [{:.15}, {:.15}]", y_final[0], y_final[1]);
        println!("  Stats: {:?}", solver.stats);
    }

    #[test]
    fn test_exponential_decay() {
        // y' = -y, y(0) = 1
        // Exact: y = exp(-t)
        struct ExpDecay;

        impl OdeSystem<1> for ExpDecay {
            fn rhs(&self, _t: f64, y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = -y[0];
            }
        }

        let sys = ExpDecay;
        let y0 = [1.0];
        let tf = 5.0;

        let tol = Tolerances::new(1e-14, 1e-14);
        let mut solver = Rkf78::new(tol);

        let (_, y_final) = solver.integrate(&sys, 0.0, &y0, tf, 0.1).unwrap();
        let exact = (-tf).exp();

        let rel_error = (y_final[0] - exact).abs() / exact;
        // Error accumulates over integration interval; 1e-11 is appropriate for tol=1e-14 over t=5
        assert!(rel_error < 1e-11, "Relative error {} too large", rel_error);

        println!("Exponential decay test passed:");
        println!("  y({}) = {:.15}, exact = {:.15}", tf, y_final[0], exact);
        println!("  Relative error: {:.3e}", rel_error);
    }

    /// Two-body problem for testing energy conservation
    struct TwoBody {
        mu: f64, // GM parameter
    }

    impl OdeSystem<6> for TwoBody {
        fn rhs(&self, _t: f64, y: &[f64; 6], dydt: &mut [f64; 6]) {
            let x = y[0];
            let y_pos = y[1];
            let z = y[2];

            let r = (x * x + y_pos * y_pos + z * z).sqrt();
            let r3 = r * r * r;
            let mu_r3 = self.mu / r3;

            // Velocity components
            dydt[0] = y[3];
            dydt[1] = y[4];
            dydt[2] = y[5];

            // Acceleration components
            dydt[3] = -mu_r3 * x;
            dydt[4] = -mu_r3 * y_pos;
            dydt[5] = -mu_r3 * z;
        }
    }

    #[test]
    fn test_two_body_energy_conservation() {
        let mu = 398600.4418; // km³/s² (Earth)
        let sys = TwoBody { mu };

        // Circular orbit at 6878 km (500 km altitude)
        let r0 = 6878.0;
        let v0 = (mu / r0).sqrt();

        // Initial state: [x, y, z, vx, vy, vz]
        let y0 = [r0, 0.0, 0.0, 0.0, v0, 0.0];

        // Integrate for one orbital period
        let period = 2.0 * std::f64::consts::PI * (r0.powi(3) / mu).sqrt();

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);

        // Compute initial energy
        let compute_energy = |y: &[f64; 6]| {
            let r = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
            let v2 = y[3] * y[3] + y[4] * y[4] + y[5] * y[5];
            0.5 * v2 - mu / r
        };

        let e0 = compute_energy(&y0);

        let (_, y_final) = solver.integrate(&sys, 0.0, &y0, period, 60.0).unwrap();

        let e_final = compute_energy(&y_final);
        let rel_energy_error = (e_final - e0).abs() / e0.abs();

        // For RKF78 with tol=1e-12, energy drift should be very small
        assert!(
            rel_energy_error < 1e-10,
            "Energy drift {} exceeds threshold",
            rel_energy_error
        );

        println!("Two-body energy conservation test passed:");
        println!("  Initial energy: {:.15e} km²/s²", e0);
        println!("  Final energy:   {:.15e} km²/s²", e_final);
        println!("  Relative drift: {:.3e}", rel_energy_error);
        println!("  Stats: {:?}", solver.stats);
    }

    #[test]
    fn test_order_of_convergence() {
        // Single-step h-refinement study on y' = cos(t), y(0) = 0, exact y = sin(t).
        // For an 8th-order method, error ~ O(h^9) per step, so
        // err(h) / err(h/2) should approach 2^9 = 512.
        // We use a broad acceptance range [100, 800] to account for
        // higher-order error terms at larger step sizes.

        struct CosODE;
        impl OdeSystem<1> for CosODE {
            fn rhs(&self, t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = t.cos();
            }
        }

        let sys = CosODE;
        let y0 = [0.0];

        // Use very loose tolerances so the solver always accepts the step
        let tol = Tolerances::new(1.0, 1.0);

        let step_sizes = [1.6, 0.8, 0.4, 0.2];
        let mut errors = Vec::new();

        for &h in &step_sizes {
            let mut solver = Rkf78::new(tol.clone());
            let result = solver.step(&sys, 0.0, &y0, h);
            assert!(result.accepted, "Step with h={} should be accepted", h);
            let exact = h.sin();
            let err = (result.y[0] - exact).abs();
            errors.push(err);
            println!(
                "h = {:.4}, y = {:.15e}, exact = {:.15e}, err = {:.3e}",
                h, result.y[0], exact, err
            );
        }

        // Check error ratios approach 2^9 = 512 (local truncation error is O(h^{p+1}))
        // Skip pairs where the smaller error is at machine epsilon (ratio meaningless)
        println!("\nError ratios (expect ~512 for 8th-order local truncation):");
        let mut checked = 0;
        for i in 0..errors.len() - 1 {
            if errors[i + 1] < 1e-15 {
                println!(
                    "  err({:.3}) / err({:.3}) — skipped (denominator at machine eps)",
                    step_sizes[i],
                    step_sizes[i + 1]
                );
                continue;
            }
            let ratio = errors[i] / errors[i + 1];
            println!(
                "  err({:.3}) / err({:.3}) = {:.1}",
                step_sizes[i],
                step_sizes[i + 1],
                ratio
            );
            assert!(
                ratio > 100.0 && ratio < 800.0,
                "Error ratio {:.1} outside [100, 800] for h={}/{}",
                ratio,
                step_sizes[i],
                step_sizes[i + 1]
            );
            checked += 1;
        }
        assert!(
            checked >= 2,
            "Need at least 2 valid error ratios, got {}",
            checked
        );
    }

    // ==================== Event Finding Tests ====================

    use crate::events::{EventAction, EventConfig, EventDirection, EventFunction};

    /// Simple event: detect when y crosses a threshold
    struct ThresholdEvent {
        threshold: f64,
    }

    impl EventFunction<1> for ThresholdEvent {
        fn eval(&self, _t: f64, y: &[f64; 1]) -> f64 {
            y[0] - self.threshold
        }
    }

    #[test]
    fn test_event_finding_exponential() {
        // y' = y, y(0) = 1, solution: y = e^t
        // Find when y = e (should be t = 1)
        struct ExpGrowth;
        impl OdeSystem<1> for ExpGrowth {
            fn rhs(&self, _t: f64, y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = y[0];
            }
        }

        let sys = ExpGrowth;
        let event = ThresholdEvent {
            threshold: std::f64::consts::E,
        };
        let config = EventConfig {
            direction: EventDirection::Rising,
            ..Default::default()
        };

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);

        let y0 = [1.0];
        let result = solver
            .integrate_to_event(&sys, &event, &config, 0.0, &y0, 10.0, 0.1)
            .unwrap();

        match result {
            IntegrationResult::Event(ev) => {
                println!("Event found at t = {:.12}", ev.t);
                println!("  y = {:.12}", ev.y[0]);
                println!("  g = {:.3e}", ev.g_value);
                println!("  iterations: {}", ev.iterations);

                // Should find t ≈ 1.0
                // Tolerance limited by linear state interpolation between steps
                assert!(
                    (ev.t - 1.0).abs() < 0.01,
                    "Event time {} should be ~1.0",
                    ev.t
                );
                assert!(
                    (ev.y[0] - std::f64::consts::E).abs() < 0.01,
                    "y should be ~e"
                );
            }
            IntegrationResult::Completed { t, .. } => {
                panic!("Expected event, but integration completed at t = {}", t);
            }
        }
    }

    #[test]
    fn test_event_finding_periapsis() {
        // Two-body orbit: detect periapsis (radial velocity = 0, rising)
        let mu = 398600.4418; // km³/s² (Earth)

        struct TwoBodyForEvent {
            mu: f64,
        }

        impl OdeSystem<6> for TwoBodyForEvent {
            fn rhs(&self, _t: f64, y: &[f64; 6], dydt: &mut [f64; 6]) {
                let r = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
                let r3 = r * r * r;
                let mu_r3 = self.mu / r3;

                dydt[0] = y[3];
                dydt[1] = y[4];
                dydt[2] = y[5];
                dydt[3] = -mu_r3 * y[0];
                dydt[4] = -mu_r3 * y[1];
                dydt[5] = -mu_r3 * y[2];
            }
        }

        // Radial velocity event (periapsis when this goes from - to +)
        struct RadialVelocityEvent;
        impl EventFunction<6> for RadialVelocityEvent {
            fn eval(&self, _t: f64, y: &[f64; 6]) -> f64 {
                // r_dot = (r · v) / |r|
                let r = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
                (y[0] * y[3] + y[1] * y[4] + y[2] * y[5]) / r
            }
        }

        let sys = TwoBodyForEvent { mu };

        // Elliptical orbit: start at apoapsis
        // Apoapsis at 42164 km (GEO), periapsis at 6678 km (300 km alt)
        let ra = 42164.0;
        let rp = 6678.0;
        let a = (ra + rp) / 2.0;
        let _e = (ra - rp) / (ra + rp);

        // Velocity at apoapsis
        let v_apo = (mu * (2.0 / ra - 1.0 / a)).sqrt();

        // Initial state at apoapsis (moving in -y direction)
        let y0 = [ra, 0.0, 0.0, 0.0, -v_apo, 0.0];

        // Orbital period
        let period = 2.0 * std::f64::consts::PI * (a.powi(3) / mu).sqrt();

        let event = RadialVelocityEvent;
        let config = EventConfig {
            direction: EventDirection::Rising, // Periapsis
            ..Default::default()
        };

        let tol = Tolerances::new(1e-10, 1e-12);
        let mut solver = Rkf78::new(tol);

        let result = solver
            .integrate_to_event(&sys, &event, &config, 0.0, &y0, period, 60.0)
            .unwrap();

        match result {
            IntegrationResult::Event(ev) => {
                let r_event = (ev.y[0].powi(2) + ev.y[1].powi(2) + ev.y[2].powi(2)).sqrt();

                println!("Periapsis found at t = {:.2} s", ev.t);
                println!("  r = {:.2} km (expected ~{:.2} km)", r_event, rp);
                println!("  g = {:.3e} (radial velocity)", ev.g_value);
                println!("  iterations: {}", ev.iterations);
                println!("  Expected time: ~{:.2} s (half period)", period / 2.0);

                // Check that we found periapsis at roughly half the orbital period
                assert!(
                    (ev.t - period / 2.0).abs() < 100.0,
                    "Periapsis time should be ~half period"
                );

                // Check that radius is close to expected periapsis
                assert!(
                    (r_event - rp).abs() < 10.0,
                    "Periapsis radius {} should be ~{} km",
                    r_event,
                    rp
                );
            }
            IntegrationResult::Completed { t, .. } => {
                panic!(
                    "Expected periapsis event, but integration completed at t = {}",
                    t
                );
            }
        }
    }

    #[test]
    fn test_no_event_reaches_tf() {
        // Simple ODE with no event occurring before tf
        struct LinearODE;
        impl OdeSystem<1> for LinearODE {
            fn rhs(&self, _t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = 1.0; // y = t + C
            }
        }

        // Event that won't trigger (y never reaches 100 before t=5)
        let event = ThresholdEvent { threshold: 100.0 };
        let config = EventConfig::default();

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);

        let y0 = [0.0];
        let result = solver
            .integrate_to_event(&LinearODE, &event, &config, 0.0, &y0, 5.0, 0.1)
            .unwrap();

        match result {
            IntegrationResult::Completed { t, y } => {
                println!("No event, completed at t = {}, y = {}", t, y[0]);
                assert!((t - 5.0).abs() < 1e-10);
                assert!((y[0] - 5.0).abs() < 1e-10);
            }
            IntegrationResult::Event(_) => {
                panic!("Should not have found an event");
            }
        }
    }

    // ==================== Phase 1: Input Validation Tests ====================

    #[test]
    fn test_nan_tolerance_rejected() {
        let tol = Tolerances::new(f64::NAN, 1e-12);
        let mut solver = Rkf78::new(tol);
        struct Dummy;
        impl OdeSystem<1> for Dummy {
            fn rhs(&self, _t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = 0.0;
            }
        }
        let result = solver.integrate(&Dummy, 0.0, &[1.0], 1.0, 0.1);
        assert!(matches!(result, Err(IntegrationError::InvalidInput { .. })));
    }

    #[test]
    fn test_inf_tolerance_rejected() {
        let tol = Tolerances::new(f64::INFINITY, 1e-12);
        let mut solver = Rkf78::new(tol);
        struct Dummy;
        impl OdeSystem<1> for Dummy {
            fn rhs(&self, _t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = 0.0;
            }
        }
        let result = solver.integrate(&Dummy, 0.0, &[1.0], 1.0, 0.1);
        assert!(matches!(result, Err(IntegrationError::InvalidInput { .. })));
    }

    #[test]
    fn test_negative_tolerance_rejected() {
        let tol = Tolerances::new(-1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);
        struct Dummy;
        impl OdeSystem<1> for Dummy {
            fn rhs(&self, _t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = 0.0;
            }
        }
        let result = solver.integrate(&Dummy, 0.0, &[1.0], 1.0, 0.1);
        assert!(matches!(result, Err(IntegrationError::InvalidInput { .. })));
    }

    #[test]
    fn test_h0_wrong_sign_rejected() {
        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);
        struct Dummy;
        impl OdeSystem<1> for Dummy {
            fn rhs(&self, _t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = 0.0;
            }
        }
        // Integrating forward but h0 is negative
        let result = solver.integrate(&Dummy, 0.0, &[1.0], 1.0, -0.1);
        assert!(matches!(result, Err(IntegrationError::InvalidInput { .. })));
    }

    #[test]
    fn test_nan_initial_state_rejected() {
        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);
        struct Dummy;
        impl OdeSystem<1> for Dummy {
            fn rhs(&self, _t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = 0.0;
            }
        }
        let result = solver.integrate(&Dummy, 0.0, &[f64::NAN], 1.0, 0.1);
        assert!(matches!(result, Err(IntegrationError::InvalidInput { .. })));
    }

    #[test]
    fn test_zero_length_integration() {
        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);
        struct Dummy;
        impl OdeSystem<1> for Dummy {
            fn rhs(&self, _t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = 1.0;
            }
        }
        let (t, y) = solver.integrate(&Dummy, 5.0, &[42.0], 5.0, 0.1).unwrap();
        assert_eq!(t, 5.0);
        assert_eq!(y[0], 42.0);
    }

    // ==================== Phase 2: Expanded Test Coverage ====================

    #[test]
    fn test_backward_integration() {
        // Harmonic oscillator integrated backward from 2π to 0
        let omega = 1.0;
        let sys = HarmonicOscillator { omega };
        let tf = 2.0 * std::f64::consts::PI;

        // Start at the known final state (should be [1, 0] after one period)
        let y0 = [1.0, 0.0];

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);

        // Integrate backward: from tf to 0, with negative step size
        let (t_final, y_final) = solver.integrate(&sys, tf, &y0, 0.0, -0.1).unwrap();

        assert!((t_final - 0.0).abs() < 1e-10, "t_final = {}", t_final);
        assert!(
            (y_final[0] - 1.0).abs() < 1e-10,
            "y(0) = {}, expected 1.0",
            y_final[0]
        );
        assert!(
            y_final[1].abs() < 1e-10,
            "y'(0) = {}, expected 0.0",
            y_final[1]
        );
    }

    #[test]
    fn test_eccentric_orbit_energy_conservation() {
        let mu = 398600.4418;
        let sys = TwoBody { mu };

        // Eccentric orbit: e=0.7, periapsis at 6678 km
        let rp = 6678.0;
        let e = 0.7;
        let a = rp / (1.0 - e);

        // Start at periapsis
        let v_peri = (mu * (2.0 / rp - 1.0 / a)).sqrt();
        let y0 = [rp, 0.0, 0.0, 0.0, v_peri, 0.0];

        let period = 2.0 * std::f64::consts::PI * (a.powi(3) / mu).sqrt();

        let compute_energy = |y: &[f64; 6]| {
            let r = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
            let v2 = y[3] * y[3] + y[4] * y[4] + y[5] * y[5];
            0.5 * v2 - mu / r
        };

        let e0 = compute_energy(&y0);

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);

        let (_, y_final) = solver.integrate(&sys, 0.0, &y0, period, 10.0).unwrap();

        let e_final = compute_energy(&y_final);
        let rel_energy_error = (e_final - e0).abs() / e0.abs();

        assert!(
            rel_energy_error < 1e-9,
            "Eccentric orbit (e=0.7) energy drift {} exceeds 1e-9",
            rel_energy_error
        );
    }

    #[test]
    fn test_hyperbolic_trajectory_energy_conservation() {
        let mu = 398600.4418;
        let sys = TwoBody { mu };

        // Hyperbolic trajectory: e=1.5, periapsis at 6678 km
        let rp = 6678.0;
        let e = 1.5;
        let a = rp / (e - 1.0); // a is positive for hyperbola in this convention

        // Start at periapsis
        let v_peri = (mu * (2.0 / rp + 1.0 / a)).sqrt();
        let y0 = [rp, 0.0, 0.0, 0.0, v_peri, 0.0];

        let compute_energy = |y: &[f64; 6]| {
            let r = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
            let v2 = y[3] * y[3] + y[4] * y[4] + y[5] * y[5];
            0.5 * v2 - mu / r
        };

        let e0 = compute_energy(&y0);
        assert!(e0 > 0.0, "Hyperbolic energy should be positive: {}", e0);

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);

        // Integrate for a reasonable time (not too long or spacecraft flies away)
        let tf = 3600.0; // 1 hour
        let (_, y_final) = solver.integrate(&sys, 0.0, &y0, tf, 10.0).unwrap();

        let e_final = compute_energy(&y_final);
        let rel_energy_error = (e_final - e0).abs() / e0.abs();

        assert!(
            rel_energy_error < 1e-9,
            "Hyperbolic trajectory energy drift {} exceeds 1e-9",
            rel_energy_error
        );
    }

    #[test]
    fn test_step_size_too_small_error() {
        // System with a singularity: y' = -1/y^2, blows up as y->0
        struct SingularODE;
        impl OdeSystem<1> for SingularODE {
            fn rhs(&self, _t: f64, y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = -1.0 / (y[0] * y[0] + 1e-30);
            }
        }

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);
        // Set h_min high enough that the step controller triggers StepSizeTooSmall
        // before we hit max_steps
        solver.h_min = 1e-4;

        // y(0) = 0.001 (start very close to singularity so step size shrinks immediately)
        let result = solver.integrate(&SingularODE, 0.0, &[0.001], 1.0, 0.0001);
        assert!(
            matches!(result, Err(IntegrationError::StepSizeTooSmall { .. })),
            "Expected StepSizeTooSmall, got {:?}",
            result
        );
    }

    #[test]
    fn test_max_steps_exceeded() {
        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);
        solver.max_steps = 5;

        let sys = HarmonicOscillator { omega: 1.0 };
        let y0 = [1.0, 0.0];

        let result = solver.integrate(&sys, 0.0, &y0, 100.0, 0.01);
        assert!(
            matches!(result, Err(IntegrationError::MaxStepsExceeded)),
            "Expected MaxStepsExceeded, got {:?}",
            result
        );
    }

    #[test]
    fn test_step_rejection_with_large_h0() {
        // Use a very large initial step size; the solver should reject steps and still converge
        let omega = 1.0;
        let sys = HarmonicOscillator { omega };
        let y0 = [1.0, 0.0];
        let tf = 2.0 * std::f64::consts::PI;

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);

        // h0 = 100 is absurdly large for this problem
        let (t_final, y_final) = solver.integrate(&sys, 0.0, &y0, tf, 100.0).unwrap();

        // Should still get the right answer
        assert!((t_final - tf).abs() < 1e-10);
        assert!(
            (y_final[0] - 1.0).abs() < 1e-9,
            "y(2π) = {}, expected 1.0",
            y_final[0]
        );

        // Should have some rejected steps
        assert!(
            solver.stats.rejected_steps > 0,
            "Expected step rejections with h0=100"
        );
    }

    #[test]
    fn test_event_near_start() {
        // y' = 1, y(0) = -0.001. Event: y = 0, should trigger very close to t = 0.001
        struct LinearGrowth;
        impl OdeSystem<1> for LinearGrowth {
            fn rhs(&self, _t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = 1.0;
            }
        }

        struct ZeroCrossing;
        impl EventFunction<1> for ZeroCrossing {
            fn eval(&self, _t: f64, y: &[f64; 1]) -> f64 {
                y[0]
            }
        }

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);

        let config = EventConfig {
            direction: EventDirection::Rising,
            ..Default::default()
        };

        let result = solver
            .integrate_to_event(
                &LinearGrowth,
                &ZeroCrossing,
                &config,
                0.0,
                &[-0.001],
                10.0,
                0.1,
            )
            .unwrap();

        match result {
            IntegrationResult::Event(ev) => {
                // Event should be near t = 0.001
                assert!(
                    (ev.t - 0.001).abs() < 0.01,
                    "Event time {} should be near 0.001",
                    ev.t
                );
            }
            IntegrationResult::Completed { .. } => {
                panic!("Expected event near start");
            }
        }
    }

    #[test]
    fn test_event_near_end() {
        // y' = 1, y(0) = 0. Event: y = 4.999. tf = 5.0
        // Event should trigger very close to tf
        struct LinearGrowth;
        impl OdeSystem<1> for LinearGrowth {
            fn rhs(&self, _t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = 1.0;
            }
        }

        let event = ThresholdEvent { threshold: 4.999 };
        let config = EventConfig {
            direction: EventDirection::Rising,
            ..Default::default()
        };

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);

        let result = solver
            .integrate_to_event(&LinearGrowth, &event, &config, 0.0, &[0.0], 5.0, 0.1)
            .unwrap();

        match result {
            IntegrationResult::Event(ev) => {
                assert!(
                    (ev.t - 4.999).abs() < 0.01,
                    "Event time {} should be near 4.999",
                    ev.t
                );
            }
            IntegrationResult::Completed { .. } => {
                panic!("Expected event near end");
            }
        }
    }

    // ==================== Phase 4: EventAction::Continue Tests ====================

    #[test]
    fn test_event_action_continue() {
        // y' = 1, y(0) = -1. Event: y = 0 (rising). With Continue, integration
        // should record the event at t ≈ 1 and keep going to tf = 5.
        struct LinearODE;
        impl OdeSystem<1> for LinearODE {
            fn rhs(&self, _t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = 1.0;
            }
        }

        struct ZeroCross;
        impl EventFunction<1> for ZeroCross {
            fn eval(&self, _t: f64, y: &[f64; 1]) -> f64 {
                y[0]
            }
        }

        let config = EventConfig {
            direction: EventDirection::Rising,
            action: EventAction::Continue,
            ..Default::default()
        };

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);

        let result = solver
            .integrate_to_event(&LinearODE, &ZeroCross, &config, 0.0, &[-1.0], 5.0, 0.1)
            .unwrap();

        // Should complete to tf (not stop at event)
        match result {
            IntegrationResult::Completed { t, y } => {
                assert!((t - 5.0).abs() < 1e-10, "Should reach tf=5, got t={}", t);
                assert!(
                    (y[0] - 4.0).abs() < 1e-10,
                    "y(5) should be 4.0, got {}",
                    y[0]
                );
            }
            IntegrationResult::Event(_) => {
                panic!("EventAction::Continue should not return Event");
            }
        }

        // Should have collected exactly 1 event
        assert_eq!(
            solver.collected_events.len(),
            1,
            "Expected 1 collected event, got {}",
            solver.collected_events.len()
        );
        let ev = &solver.collected_events[0];
        assert!(
            (ev.t - 1.0).abs() < 0.01,
            "Event time {} should be near 1.0",
            ev.t
        );
    }

    // ==================== Long-Duration & Round-Trip Tests ====================

    #[test]
    fn test_100_orbit_energy_conservation() {
        let mu = 398600.4418;
        let sys = TwoBody { mu };

        let r0 = 6878.0;
        let v0 = (mu / r0).sqrt();
        let y0 = [r0, 0.0, 0.0, 0.0, v0, 0.0];

        let period = 2.0 * std::f64::consts::PI * (r0.powi(3) / mu).sqrt();
        let tf = 100.0 * period;

        let compute_energy = |y: &[f64; 6]| {
            let r = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
            let v2 = y[3] * y[3] + y[4] * y[4] + y[5] * y[5];
            0.5 * v2 - mu / r
        };

        let e0 = compute_energy(&y0);

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);

        let (_, y_final) = solver.integrate(&sys, 0.0, &y0, tf, 60.0).unwrap();
        let e_final = compute_energy(&y_final);
        let rel_energy_error = (e_final - e0).abs() / e0.abs();

        println!("100-orbit energy drift: {:.3e}", rel_energy_error);
        assert!(
            rel_energy_error < 1e-8,
            "100-orbit energy drift {} exceeds 1e-8",
            rel_energy_error
        );
    }

    #[test]
    fn test_forward_backward_round_trip() {
        let sys = HarmonicOscillator { omega: 1.0 };
        let y0 = [1.0, 0.0];
        let period = 2.0 * std::f64::consts::PI;

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol.clone());

        // Forward one period
        let (t_mid, y_mid) = solver.integrate(&sys, 0.0, &y0, period, 0.1).unwrap();

        // Backward one period
        let mut solver2 = Rkf78::new(tol);
        let (t_final, y_final) = solver2.integrate(&sys, t_mid, &y_mid, 0.0, -0.1).unwrap();

        assert!(
            t_final.abs() < 1e-10,
            "Round-trip t = {}, expected 0",
            t_final
        );
        assert!(
            (y_final[0] - y0[0]).abs() < 1e-10,
            "Round-trip y[0] = {}, expected {}",
            y_final[0],
            y0[0]
        );
        assert!(
            (y_final[1] - y0[1]).abs() < 1e-10,
            "Round-trip y[1] = {}, expected {}",
            y_final[1],
            y0[1]
        );
    }

    #[test]
    fn test_per_component_tolerance() {
        // Verify that per-component tolerances work with different-scale components.
        // Harmonic oscillator: y[0] = cos(t), y[1] = -sin(t).
        // Use tight atol for position (large ~1), loose atol for velocity (also ~1),
        // and compare step counts: tighter uniform tolerance needs more steps.
        let sys = HarmonicOscillator { omega: 1.0 };
        let y0 = [1.0, 0.0];
        let tf = 10.0 * std::f64::consts::PI;

        // Run with loose uniform tolerance
        let tol_loose = Tolerances::new(1e-6, 1e-6);
        let mut solver_loose = Rkf78::new(tol_loose);
        let (_, y_loose) = solver_loose.integrate(&sys, 0.0, &y0, tf, 0.1).unwrap();
        let steps_loose = solver_loose.stats.accepted_steps;

        // Run with tight uniform tolerance
        let tol_tight = Tolerances::new(1e-13, 1e-13);
        let mut solver_tight = Rkf78::new(tol_tight);
        let (_, y_tight) = solver_tight.integrate(&sys, 0.0, &y0, tf, 0.1).unwrap();
        let steps_tight = solver_tight.stats.accepted_steps;

        // Run with per-component: tight on y[0], loose on y[1]
        let tol_mixed = Tolerances::with_components([1e-13, 1e-6], [1e-13, 1e-6]);
        let mut solver_mixed = Rkf78::new(tol_mixed);
        let (_, y_mixed) = solver_mixed.integrate(&sys, 0.0, &y0, tf, 0.1).unwrap();
        let steps_mixed = solver_mixed.stats.accepted_steps;

        println!(
            "Steps: loose={}, mixed={}, tight={}",
            steps_loose, steps_mixed, steps_tight
        );

        // Mixed tolerance should need more steps than loose (tight y[0] drives step size)
        assert!(
            steps_mixed > steps_loose,
            "Per-component tight should need more steps ({}) than loose ({})",
            steps_mixed,
            steps_loose
        );

        // y[0] accuracy with mixed should be close to tight (since y[0] drives step size)
        let exact_y0 = tf.cos();
        let err_tight = (y_tight[0] - exact_y0).abs();
        let err_mixed = (y_mixed[0] - exact_y0).abs();
        let err_loose = (y_loose[0] - exact_y0).abs();

        println!(
            "y[0] errors: loose={:.3e}, mixed={:.3e}, tight={:.3e}",
            err_loose, err_mixed, err_tight
        );

        // Mixed should be much better than loose for y[0]
        assert!(
            err_mixed < err_loose || err_loose < 1e-10,
            "Per-component should improve accuracy of tight component"
        );
    }

    // ==================== Step Controller Boundary Tests ====================

    #[test]
    fn test_step_controller_zero_error() {
        let ctrl = StepController::default();
        let factor = ctrl.compute_factor(0.0);
        assert_eq!(factor, ctrl.max_factor, "error=0 should give max_factor");
    }

    #[test]
    fn test_step_controller_unit_error() {
        let ctrl = StepController::default();
        let factor = ctrl.compute_factor(1.0);
        // safety * 1.0^(-1/8) = 0.9 * 1.0 = 0.9
        assert!(
            (factor - ctrl.safety).abs() < 1e-15,
            "error=1.0 should give safety={}, got {}",
            ctrl.safety,
            factor
        );
    }

    #[test]
    fn test_step_controller_tiny_error_clamped() {
        let ctrl = StepController::default();
        let factor = ctrl.compute_factor(1e-20);
        assert_eq!(
            factor, ctrl.max_factor,
            "very small error should clamp to max_factor"
        );
    }

    #[test]
    fn test_step_controller_huge_error_clamped() {
        let ctrl = StepController::default();
        let factor = ctrl.compute_factor(1e+20);
        assert_eq!(
            factor, ctrl.min_factor,
            "very large error should clamp to min_factor"
        );
    }

    #[test]
    fn test_tolerance_sensitivity() {
        // Harmonic oscillator over 10 periods: tighter tolerances should give smaller errors.
        let sys = HarmonicOscillator { omega: 1.0 };
        let y0 = [1.0, 0.0];
        let tf = 10.0 * 2.0 * std::f64::consts::PI;

        let exact_y0 = tf.cos();

        let run = |atol: f64, rtol: f64| -> f64 {
            let tol = Tolerances::new(atol, rtol);
            let mut solver = Rkf78::new(tol);
            let (_, y_final) = solver.integrate(&sys, 0.0, &y0, tf, 0.1).unwrap();
            (y_final[0] - exact_y0).abs()
        };

        let err_loose = run(1e-8, 1e-8);
        let err_medium = run(1e-10, 1e-10);
        let err_tight = run(1e-12, 1e-12);

        println!(
            "Tolerance sensitivity: loose={:.3e}, medium={:.3e}, tight={:.3e}",
            err_loose, err_medium, err_tight
        );

        assert!(
            err_loose > err_medium,
            "Loose error {:.3e} should exceed medium {:.3e}",
            err_loose,
            err_medium
        );
        assert!(
            err_medium > err_tight,
            "Medium error {:.3e} should exceed tight {:.3e}",
            err_medium,
            err_tight
        );
    }

    #[test]
    fn test_high_eccentricity_orbit_energy() {
        // High-eccentricity orbit (e=0.99): energy conservation over one full period.
        let mu = 398600.4418;
        let sys = TwoBody { mu };

        let rp = 6678.0; // 300 km periapsis
        let e = 0.99;
        let a = rp / (1.0 - e);

        let v_peri = (mu * (2.0 / rp - 1.0 / a)).sqrt();
        let y0 = [rp, 0.0, 0.0, 0.0, v_peri, 0.0];

        let period = 2.0 * std::f64::consts::PI * (a.powi(3) / mu).sqrt();

        let compute_energy = |y: &[f64; 6]| {
            let r = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
            let v2 = y[3] * y[3] + y[4] * y[4] + y[5] * y[5];
            0.5 * v2 - mu / r
        };

        let e0 = compute_energy(&y0);

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);

        let (_, y_final) = solver.integrate(&sys, 0.0, &y0, period, 1.0).unwrap();

        let e_final = compute_energy(&y_final);
        let rel_energy_error = (e_final - e0).abs() / e0.abs();

        println!(
            "High-eccentricity (e=0.99) energy drift: {:.3e} (period = {:.0} s)",
            rel_energy_error, period
        );

        // High-e orbits are challenging; 1e-6 is a reasonable threshold
        assert!(
            rel_energy_error < 1e-6,
            "High-e orbit energy drift {} exceeds 1e-6",
            rel_energy_error
        );
    }

    #[test]
    fn test_event_action_continue_multiple() {
        // Harmonic oscillator: y = cos(t), y' = -sin(t)
        // Event: y[0] = 0 (any direction). Zeros at π/2, 3π/2, 5π/2, 7π/2 in [0, 4π]
        let sys = HarmonicOscillator { omega: 1.0 };

        struct PositionZero;
        impl EventFunction<2> for PositionZero {
            fn eval(&self, _t: f64, y: &[f64; 2]) -> f64 {
                y[0]
            }
        }

        let config = EventConfig {
            direction: EventDirection::Any,
            action: EventAction::Continue,
            ..Default::default()
        };

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);

        let tf = 4.0 * std::f64::consts::PI;
        let result = solver
            .integrate_to_event(&sys, &PositionZero, &config, 0.0, &[1.0, 0.0], tf, 0.1)
            .unwrap();

        // Should complete to tf
        match result {
            IntegrationResult::Completed { t, .. } => {
                assert!((t - tf).abs() < 1e-10, "Should reach tf, got t={}", t);
            }
            IntegrationResult::Event(_) => {
                panic!("EventAction::Continue should not return Event");
            }
        }

        // cos(t) = 0 at t = π/2, 3π/2, 5π/2, 7π/2 → 4 crossings in [0, 4π]
        assert!(
            solver.collected_events.len() >= 4,
            "Expected at least 4 zero-crossings, got {}",
            solver.collected_events.len()
        );

        // Verify the first few event times are near the expected zeros
        let pi = std::f64::consts::PI;
        let expected_times = [pi / 2.0, 3.0 * pi / 2.0, 5.0 * pi / 2.0, 7.0 * pi / 2.0];
        for (i, expected_t) in expected_times.iter().enumerate() {
            if i < solver.collected_events.len() {
                let actual_t = solver.collected_events[i].t;
                assert!(
                    (actual_t - expected_t).abs() < 0.05,
                    "Event {} at t={:.4}, expected {:.4}",
                    i,
                    actual_t,
                    expected_t
                );
            }
        }
    }
}
