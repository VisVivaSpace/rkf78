//! Runge-Kutta-Fehlberg 7(8) Integrator
//!
//! A 13-stage embedded RK7(8) pair for high-precision integration of ODEs.
//! Designed for spacecraft trajectory propagation and astrodynamics applications.
//!
//! Reference: NASA TR R-287, Erwin Fehlberg, 1968

use crate::coefficients::{A, B, B_ERR, C, STAGES};
use crate::events::{
    hermite_interp, sign_change_detected, BrentError, BrentSolver, EventAction, EventConfig,
    EventFunction, EventResult,
};
use crate::scalar::{Float, Scalar};

/// System of ordinary differential equations: dy/dt = f(t, y)
pub trait OdeSystem<T: Scalar, const N: usize> {
    /// Evaluate the right-hand side of the ODE system
    ///
    /// # Arguments
    /// * `t` - Current time
    /// * `y` - Current state vector
    /// * `dydt` - Output: derivative dy/dt
    fn rhs(&self, t: T::Real, y: &[T; N], dydt: &mut [T; N]);
}

/// Observer called after each accepted integration step.
///
/// Implement this trait to inspect the integration trajectory without
/// storing every step. The observer is called only for accepted steps.
///
/// # Example
///
/// ```ignore
/// use rkf78::{StepObserver, Scalar};
///
/// struct StepCounter(u64);
///
/// impl<T: Scalar, const N: usize> StepObserver<T, N> for StepCounter {
///     fn on_step(&mut self, _t: T::Real, _y: &[T; N], _h: T::Real, _error: T::Real) {
///         self.0 += 1;
///     }
/// }
/// ```
pub trait StepObserver<T: Scalar, const N: usize> {
    /// Called after each accepted step.
    ///
    /// # Arguments
    /// * `t` - Time after the step
    /// * `y` - State after the step
    /// * `h` - Step size used (signed)
    /// * `error` - Normalized error estimate for this step
    fn on_step(&mut self, t: T::Real, y: &[T; N], h: T::Real, error: T::Real);
}

/// No-op observer — used by `integrate()` to avoid overhead.
impl<T: Scalar, const N: usize> StepObserver<T, N> for () {
    #[inline]
    fn on_step(&mut self, _t: T::Real, _y: &[T; N], _h: T::Real, _error: T::Real) {}
}

/// Integration result from a single step
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct StepResult<T: Scalar, const N: usize> {
    /// New state after the step (8th order solution)
    pub y: [T; N],
    /// New time value
    pub t: T::Real,
    /// Normalized error estimate (should be <= 1.0 for acceptance)
    pub error: T::Real,
    /// Suggested step size for next step
    pub h_next: T::Real,
    /// Whether the step was accepted
    pub accepted: bool,
}

/// Integration statistics for diagnostics
#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    /// Total number of function evaluations
    pub fn_evals: u64,
    /// Number of accepted steps
    pub accepted_steps: u64,
    /// Number of rejected steps
    pub rejected_steps: u64,
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "fn_evals: {}, accepted: {}, rejected: {}",
            self.fn_evals, self.accepted_steps, self.rejected_steps
        )
    }
}

/// Step-size controller using an I-controller
///
/// h_new = safety * h * error^(-1/p)
/// where p = 8 for RKF78
#[derive(Clone, Copy)]
pub struct StepController<R: Float> {
    /// Safety factor (0.8-0.9 typical)
    pub safety: R,
    /// Maximum growth factor per step
    pub max_factor: R,
    /// Minimum reduction factor per step
    pub min_factor: R,
    /// Exponent = 1/(order + 1) for I-controller
    exponent: R,
}

impl<R: Float> Default for StepController<R> {
    fn default() -> Self {
        Self {
            safety: R::from_f64(0.9),
            max_factor: R::from_f64(5.0),
            min_factor: R::from_f64(0.2),
            exponent: R::from_f64(1.0 / 8.0), // 1/(p+1) where p=7 for error estimate order
        }
    }
}

impl<R: Float> StepController<R> {
    /// Create a new step controller with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the safety factor (0.8–0.9 typical).
    pub fn with_safety(mut self, v: R) -> Self {
        self.safety = v;
        self
    }

    /// Set the maximum growth factor per step.
    pub fn with_max_factor(mut self, v: R) -> Self {
        self.max_factor = v;
        self
    }

    /// Set the minimum reduction factor per step.
    pub fn with_min_factor(mut self, v: R) -> Self {
        self.min_factor = v;
        self
    }

    /// Compute the step size adjustment factor
    pub fn compute_factor(&self, error: R) -> R {
        if error == R::ZERO {
            return self.max_factor;
        }

        let factor = self.safety * error.powf(-self.exponent);
        factor.clamp(self.min_factor, self.max_factor)
    }
}

/// Tolerance specification for error control
///
/// Error is computed as: |y8 - y7| / (atol + rtol * |y8|)
#[derive(Debug, Clone, Copy)]
pub struct Tolerances<R: Float, const N: usize> {
    /// Absolute tolerance per component
    pub atol: [R; N],
    /// Relative tolerance per component
    pub rtol: [R; N],
}

impl<R: Float, const N: usize> Tolerances<R, N> {
    /// Create tolerances with uniform values
    pub fn new(atol: R, rtol: R) -> Self {
        Self {
            atol: [atol; N],
            rtol: [rtol; N],
        }
    }

    /// Create tolerances with per-component values
    pub fn with_components(atol: [R; N], rtol: [R; N]) -> Self {
        Self { atol, rtol }
    }
}

/// Configuration for an integration run.
///
/// Specifies the time span, initial step size, step limits, and max steps.
/// The `h0` parameter is always a positive magnitude — integration direction
/// is inferred from `(tf - t0).signum()`.
#[derive(Debug, Clone, Copy)]
pub struct IntegrationConfig<R: Float> {
    /// Initial time
    pub t0: R,
    /// Final time
    pub tf: R,
    /// Initial step size (positive magnitude; direction inferred from tf − t0)
    pub h0: R,
    /// Minimum step size magnitude (default: 1e-14)
    pub h_min: R,
    /// Maximum step size magnitude (default: infinity)
    pub h_max: R,
    /// Maximum number of integration steps (default: 10,000,000)
    pub max_steps: u64,
}

impl<R: Float> IntegrationConfig<R> {
    /// Create a new integration config.
    ///
    /// `h0` is stored as its absolute value. Integration direction
    /// is inferred from `tf - t0`.
    pub fn new(t0: R, tf: R, h0: R) -> Self {
        Self {
            t0,
            tf,
            h0: h0.abs(),
            h_min: R::from_f64(1e-14),
            h_max: R::INFINITY,
            max_steps: 10_000_000,
        }
    }

    /// Set the minimum step size magnitude.
    pub fn with_h_min(mut self, v: R) -> Self {
        self.h_min = v;
        self
    }

    /// Set the maximum step size magnitude.
    pub fn with_h_max(mut self, v: R) -> Self {
        self.h_max = v;
        self
    }

    /// Set the maximum number of integration steps.
    pub fn with_max_steps(mut self, v: u64) -> Self {
        self.max_steps = v;
        self
    }
}

/// Runge-Kutta-Fehlberg 7(8) integrator
///
/// # Type Parameters
/// * `T` - Scalar type for state components (f32, f64, or complex)
/// * `N` - Dimension of the state vector
///
/// # Example
/// ```ignore
/// use rkf78::{Rkf78, OdeSystem, Tolerances, IntegrationConfig};
///
/// struct HarmonicOscillator { omega: f64 }
///
/// impl OdeSystem<f64, 2> for HarmonicOscillator {
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
/// let config = IntegrationConfig::new(0.0, 10.0, 0.1);
///
/// let (tf, yf) = solver.integrate(&sys, &config, &y0).unwrap();
/// ```
#[derive(Clone)]
pub struct Rkf78<T: Scalar, const N: usize> {
    /// Tolerance specification
    tol: Tolerances<T::Real, N>,
    /// Step-size controller
    controller: StepController<T::Real>,
    /// Stage evaluations (pre-allocated workspace)
    k: [[T; N]; STAGES],
    /// Integration statistics
    pub stats: Stats,
}

impl<T: Scalar, const N: usize> Rkf78<T, N> {
    /// Create a new RKF78 solver with specified tolerances
    pub fn new(tol: Tolerances<T::Real, N>) -> Self {
        Self {
            tol,
            controller: StepController::default(),
            k: [[T::ZERO; N]; STAGES],
            stats: Stats::default(),
        }
    }

    /// Set a custom step-size controller.
    pub fn with_controller(mut self, controller: StepController<T::Real>) -> Self {
        self.controller = controller;
        self
    }

    /// Perform a single integration step
    ///
    /// This computes the 13 stages, forms the 8th and 7th order solutions,
    /// estimates the error, and determines if the step should be accepted.
    /// Perform a single integration step
    ///
    /// This computes the 13 stages, forms the 8th and 7th order solutions,
    /// estimates the error, and determines if the step should be accepted.
    ///
    /// `h` is used as-is (signed). The returned `h_next` is a positive
    /// magnitude — the caller is responsible for applying direction and
    /// clamping to step-size limits.
    pub fn step<S: OdeSystem<T, N>>(
        &mut self,
        sys: &S,
        t: T::Real,
        y: &[T; N],
        h: T::Real,
    ) -> StepResult<T, N> {
        // Compute all 13 stages
        self.compute_stages(sys, t, y, h);

        // Compute 8th order solution
        let y8 = self.compute_solution(y, h);

        // Compute error estimate
        let error = self.compute_error(&y8, h);

        // Determine acceptance
        let accepted = error <= T::Real::ONE;

        // Compute next step size (positive magnitude, no h_min/h_max clamping)
        let factor = self.controller.compute_factor(error);
        let h_next = h.abs() * factor;

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

    /// Integrate from `config.t0` to `config.tf`.
    ///
    /// # Arguments
    /// * `sys` - The ODE system to integrate
    /// * `config` - Integration configuration (time span, step size, limits)
    /// * `y0` - Initial state
    ///
    /// # Returns
    /// * `Ok((t_final, y_final))` on success
    /// * `Err(IntegrationError)` on failure
    #[must_use = "integration result contains the final state"]
    #[allow(clippy::type_complexity)]
    pub fn integrate<S: OdeSystem<T, N>>(
        &mut self,
        sys: &S,
        config: &IntegrationConfig<T::Real>,
        y0: &[T; N],
    ) -> Result<(T::Real, [T; N]), IntegrationError<T::Real>> {
        self.integrate_with_observer(sys, config, y0, &mut ())
    }

    /// Integrate from `config.t0` to `config.tf`, calling `observer` after
    /// each accepted step.
    ///
    /// This is identical to [`integrate()`](Self::integrate) but invokes
    /// `observer.on_step(t, y, h, error)` after every accepted step. Use
    /// this to record trajectories, compute running statistics, or implement
    /// custom termination logic (via panicking — prefer events for that).
    ///
    /// # Arguments
    /// * `sys` - The ODE system to integrate
    /// * `config` - Integration configuration (time span, step size, limits)
    /// * `y0` - Initial state
    /// * `observer` - Called after each accepted step
    ///
    /// # Returns
    /// * `Ok((t_final, y_final))` on success
    /// * `Err(IntegrationError)` on failure
    #[must_use = "integration result contains the final state"]
    #[allow(clippy::type_complexity)]
    pub fn integrate_with_observer<S, O>(
        &mut self,
        sys: &S,
        config: &IntegrationConfig<T::Real>,
        y0: &[T; N],
        observer: &mut O,
    ) -> Result<(T::Real, [T; N]), IntegrationError<T::Real>>
    where
        S: OdeSystem<T, N>,
        O: StepObserver<T, N>,
    {
        if config.t0 == config.tf {
            return Ok((config.t0, *y0));
        }
        self.validate_inputs(config, y0)?;

        let mut t = config.t0;
        let mut y = *y0;
        let direction = (config.tf - config.t0).signum();
        let mut h = config.h0.clamp(config.h_min, config.h_max) * direction;

        let mut step_count = 0u64;

        while (config.tf - t) * direction > config.h_min {
            // Don't overshoot the endpoint
            if (t + h - config.tf) * direction > T::Real::ZERO {
                h = config.tf - t;
            }

            let result = self.step(sys, t, &y, h);

            if result.accepted {
                t = result.t;
                y = result.y;
                if !y.iter().all(|v| v.norm().is_finite()) {
                    return Err(IntegrationError::NonFiniteState { t });
                }
                observer.on_step(t, &y, h, result.error);
            }

            h = result.h_next.clamp(config.h_min, config.h_max) * direction;

            step_count += 1;
            if step_count > config.max_steps {
                return Err(IntegrationError::MaxStepsExceeded);
            }

            // Check for step size too small: if the step was rejected and
            // the next step size is already at h_min, we can't make progress
            if !result.accepted
                && result.h_next <= config.h_min
                && (config.tf - t) * direction > config.h_min
            {
                return Err(IntegrationError::StepSizeTooSmall {
                    t,
                    h: result.h_next,
                });
            }
        }

        Ok((t, y))
    }

    /// Integrate from `config.t0` to `config.tf`, recording the full trajectory.
    ///
    /// Returns a [`Solution`] that supports Hermite cubic interpolation at
    /// arbitrary times within the integration span, plus the final `(t, y)`.
    ///
    /// **Cost:** One extra RHS evaluation per accepted step (to obtain the
    /// derivative at the step endpoint for Hermite interpolation). This is
    /// approximately 7-8% overhead compared to [`integrate()`](Self::integrate).
    ///
    /// # Arguments
    /// * `sys` - The ODE system to integrate
    /// * `config` - Integration configuration (time span, step size, limits)
    /// * `y0` - Initial state
    ///
    /// # Returns
    /// * `Ok((t_final, y_final, solution))` on success
    /// * `Err(IntegrationError)` on failure
    #[must_use = "integration result contains the trajectory"]
    #[allow(clippy::type_complexity)]
    pub fn integrate_dense<S: OdeSystem<T, N>>(
        &mut self,
        sys: &S,
        config: &IntegrationConfig<T::Real>,
        y0: &[T; N],
    ) -> Result<(T::Real, [T; N], crate::Solution<T, N>), IntegrationError<T::Real>> {
        if config.t0 == config.tf {
            let mut sol = crate::Solution::with_capacity(1);
            let mut f0 = [T::ZERO; N];
            sys.rhs(config.t0, y0, &mut f0);
            self.stats.fn_evals += 1;
            sol.push(config.t0, *y0, f0);
            return Ok((config.t0, *y0, sol));
        }
        self.validate_inputs(config, y0)?;

        let mut t = config.t0;
        let mut y = *y0;
        let direction = (config.tf - config.t0).signum();
        let mut h = config.h0.clamp(config.h_min, config.h_max) * direction;

        // Record initial point
        let mut sol = crate::Solution::with_capacity(128);
        let mut f = [T::ZERO; N];
        sys.rhs(t, &y, &mut f);
        self.stats.fn_evals += 1;
        sol.push(t, y, f);

        let mut step_count = 0u64;

        while (config.tf - t) * direction > config.h_min {
            if (t + h - config.tf) * direction > T::Real::ZERO {
                h = config.tf - t;
            }

            let result = self.step(sys, t, &y, h);

            if result.accepted {
                t = result.t;
                y = result.y;
                if !y.iter().all(|v| v.norm().is_finite()) {
                    return Err(IntegrationError::NonFiniteState { t });
                }
                // Compute derivative at new point for Hermite interpolation
                sys.rhs(t, &y, &mut f);
                self.stats.fn_evals += 1;
                sol.push(t, y, f);
            }

            h = result.h_next.clamp(config.h_min, config.h_max) * direction;

            step_count += 1;
            if step_count > config.max_steps {
                return Err(IntegrationError::MaxStepsExceeded);
            }

            if !result.accepted
                && result.h_next <= config.h_min
                && (config.tf - t) * direction > config.h_min
            {
                return Err(IntegrationError::StepSizeTooSmall {
                    t,
                    h: result.h_next,
                });
            }
        }

        Ok((t, y, sol))
    }

    /// Compute all 13 stages
    #[allow(clippy::needless_range_loop)]
    fn compute_stages<S: OdeSystem<T, N>>(&mut self, sys: &S, t: T::Real, y: &[T; N], h: T::Real) {
        let mut y_temp = [T::ZERO; N];

        // Stage 0: k[0] = f(t, y)
        sys.rhs(t, y, &mut self.k[0]);

        // Stages 1-12
        for i in 1..STAGES {
            // y_temp = y + h * sum_{j=0}^{i-1} a[i][j] * k[j]
            for n in 0..N {
                let mut sum = T::ZERO;
                for j in 0..i {
                    sum += self.k[j][n].mul_real(T::Real::from_f64(A[i][j]));
                }
                y_temp[n] = y[n] + sum.mul_real(h);
            }

            // k[i] = f(t + c[i]*h, y_temp)
            sys.rhs(t + T::Real::from_f64(C[i]) * h, &y_temp, &mut self.k[i]);
        }
    }

    /// Compute the 8th order solution from the stages
    #[allow(clippy::needless_range_loop)]
    fn compute_solution(&self, y: &[T; N], h: T::Real) -> [T; N] {
        let mut y_new = [T::ZERO; N];

        for n in 0..N {
            let mut sum = T::ZERO;
            for i in 0..STAGES {
                sum += self.k[i][n].mul_real(T::Real::from_f64(B[i]));
            }
            y_new[n] = y[n] + sum.mul_real(h);
        }

        y_new
    }

    /// Compute the normalized error estimate
    ///
    /// Uses the infinity norm of the scaled error:
    /// error = max_i( |h * sum_j (b[j] - b_hat[j]) * k[j][i]| / scale[i] )
    /// where scale[i] = atol[i] + rtol[i] * |y8[i]|
    #[allow(clippy::needless_range_loop)]
    fn compute_error(&self, y8: &[T; N], h: T::Real) -> T::Real {
        let mut max_err = T::Real::ZERO;

        for n in 0..N {
            // Compute error in component n
            let mut err_sum = T::ZERO;
            for i in 0..STAGES {
                err_sum += self.k[i][n].mul_real(T::Real::from_f64(B_ERR[i]));
            }
            let err_val = err_sum.mul_real(h);

            // Scale by tolerance
            let scale = self.tol.atol[n] + self.tol.rtol[n] * y8[n].norm();
            let scaled_err = err_val.norm() / scale;

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
        config: &IntegrationConfig<T::Real>,
        y0: &[T; N],
    ) -> Result<(), IntegrationError<T::Real>> {
        if !config.t0.is_finite() || !config.tf.is_finite() || !config.h0.is_finite() {
            return Err(IntegrationError::InvalidInput {
                message: "t0, tf, and h0 must be finite".to_string(),
            });
        }
        if config.h0 == T::Real::ZERO {
            return Err(IntegrationError::InvalidInput {
                message: "h0 must be non-zero".to_string(),
            });
        }
        for (i, val) in y0.iter().enumerate() {
            if !val.norm().is_finite() {
                return Err(IntegrationError::InvalidInput {
                    message: format!("y0[{}] is not finite", i),
                });
            }
        }
        for (i, (&a, &r)) in self.tol.atol.iter().zip(self.tol.rtol.iter()).enumerate() {
            if !a.is_finite() || a <= T::Real::ZERO {
                return Err(IntegrationError::InvalidInput {
                    message: format!("atol[{}] must be positive and finite", i),
                });
            }
            if !r.is_finite() || r < T::Real::ZERO {
                return Err(IntegrationError::InvalidInput {
                    message: format!("rtol[{}] must be non-negative and finite", i),
                });
            }
        }
        Ok(())
    }

    /// Integrate with a single event function.
    ///
    /// Monitors event function `g(t, y)` during integration. When `g` changes
    /// sign, Brent's method locates the precise crossing time.
    ///
    /// **Note:** The event state is found via Hermite cubic interpolation
    /// between integration steps, giving O(h^4) accuracy in the event state.
    ///
    /// # Returns
    /// * `Ok((IntegrationResult, collected_events))` — The first element is
    ///   either `Event(...)` (if a `Stop` event fired) or `Completed { t, y }`.
    ///   The second element contains all `Continue`-action events recorded
    ///   during integration (empty if no `Continue` events occurred).
    #[must_use = "integration result contains the final state or event"]
    #[allow(clippy::type_complexity)]
    pub fn integrate_to_event<S, E>(
        &mut self,
        sys: &S,
        event: &E,
        event_config: &EventConfig<T::Real>,
        config: &IntegrationConfig<T::Real>,
        y0: &[T; N],
    ) -> Result<(IntegrationResult<T, N>, Vec<EventResult<T, N>>), IntegrationError<T::Real>>
    where
        S: OdeSystem<T, N>,
        E: EventFunction<T, N>,
    {
        if config.t0 == config.tf {
            return Ok((
                IntegrationResult::Completed {
                    t: config.t0,
                    y: *y0,
                },
                Vec::new(),
            ));
        }
        self.validate_inputs(config, y0)?;

        let mut t = config.t0;
        let mut y = *y0;
        let direction = (config.tf - config.t0).signum();
        let mut h = config.h0.clamp(config.h_min, config.h_max) * direction;
        let mut collected = Vec::new();

        // Evaluate initial event function
        let mut g_prev = event.eval(t, &y);

        let mut step_count = 0u64;

        while (config.tf - t) * direction > config.h_min {
            if (t + h - config.tf) * direction > T::Real::ZERO {
                h = config.tf - t;
            }

            let result = self.step(sys, t, &y, h);

            if result.accepted {
                let g_new = event.eval(result.t, &result.y);

                if sign_change_detected(g_prev, g_new, event_config.direction) {
                    // Compute Hermite data for root-finding
                    let mut f_a = [T::ZERO; N];
                    let mut f_b = [T::ZERO; N];
                    sys.rhs(t, &y, &mut f_a);
                    sys.rhs(result.t, &result.y, &mut f_b);
                    self.stats.fn_evals += 2;

                    let (t_a, y_a, t_b, y_b) = (t, y, result.t, result.y);
                    let event_result = Self::find_root_brent(
                        t_a,
                        t_b,
                        &y_a,
                        &y_b,
                        &f_a,
                        &f_b,
                        g_prev,
                        g_new,
                        0,
                        event_config,
                        |ti| {
                            let yi = hermite_interp(t_a, t_b, &y_a, &y_b, &f_a, &f_b, ti);
                            event.eval(ti, &yi)
                        },
                    )?;

                    match event_config.action {
                        EventAction::Stop => {
                            return Ok((IntegrationResult::Event(event_result), collected));
                        }
                        EventAction::Continue => {
                            collected.push(event_result);
                            t = result.t;
                            y = result.y;
                            g_prev = g_new;
                            h = result.h_next.clamp(config.h_min, config.h_max) * direction;
                            continue;
                        }
                    }
                }

                t = result.t;
                y = result.y;
                if !y.iter().all(|v| v.norm().is_finite()) {
                    return Err(IntegrationError::NonFiniteState { t });
                }
                g_prev = g_new;
            }

            h = result.h_next.clamp(config.h_min, config.h_max) * direction;

            step_count += 1;
            if step_count > config.max_steps {
                return Err(IntegrationError::MaxStepsExceeded);
            }

            if !result.accepted
                && result.h_next <= config.h_min
                && (config.tf - t) * direction > config.h_min
            {
                return Err(IntegrationError::StepSizeTooSmall {
                    t,
                    h: result.h_next,
                });
            }
        }

        Ok((IntegrationResult::Completed { t, y }, collected))
    }

    /// Integrate with M simultaneous event functions.
    ///
    /// Monitors M event functions simultaneously. When any event changes sign,
    /// Brent's method locates each crossing independently, and the earliest
    /// one is processed. Each event has its own [`EventConfig`].
    ///
    /// # Returns
    /// * `Ok((IntegrationResult, collected_events))` — same as
    ///   [`integrate_to_event`](Self::integrate_to_event), with
    ///   [`EventResult::event_index`] identifying which event fired.
    #[must_use = "integration result contains the final state or event"]
    #[allow(clippy::type_complexity)]
    pub fn integrate_with_multi_events<S, E, const M: usize>(
        &mut self,
        sys: &S,
        events: &E,
        event_configs: &[EventConfig<T::Real>; M],
        config: &IntegrationConfig<T::Real>,
        y0: &[T; N],
    ) -> Result<(IntegrationResult<T, N>, Vec<EventResult<T, N>>), IntegrationError<T::Real>>
    where
        S: OdeSystem<T, N>,
        E: crate::events::MultiEventFunction<T, N, M>,
    {
        if config.t0 == config.tf {
            return Ok((
                IntegrationResult::Completed {
                    t: config.t0,
                    y: *y0,
                },
                Vec::new(),
            ));
        }
        self.validate_inputs(config, y0)?;

        let mut t = config.t0;
        let mut y = *y0;
        let direction = (config.tf - config.t0).signum();
        let mut h = config.h0.clamp(config.h_min, config.h_max) * direction;
        let mut collected = Vec::new();

        let mut g_prev = events.eval(t, &y);

        let mut step_count = 0u64;

        while (config.tf - t) * direction > config.h_min {
            if (t + h - config.tf) * direction > T::Real::ZERO {
                h = config.tf - t;
            }

            let result = self.step(sys, t, &y, h);

            if result.accepted {
                let g_new = events.eval(result.t, &result.y);

                // Check all M events for sign changes, find earliest root
                let mut earliest: Option<EventResult<T, N>> = None;
                let mut earliest_action = EventAction::Stop;
                let mut needs_hermite = false;

                for m in 0..M {
                    if sign_change_detected(g_prev[m], g_new[m], event_configs[m].direction) {
                        needs_hermite = true;
                        break;
                    }
                }

                if needs_hermite {
                    let mut f_a = [T::ZERO; N];
                    let mut f_b = [T::ZERO; N];
                    sys.rhs(t, &y, &mut f_a);
                    sys.rhs(result.t, &result.y, &mut f_b);
                    self.stats.fn_evals += 2;

                    let (t_a, y_a, t_b, y_b) = (t, y, result.t, result.y);

                    for m in 0..M {
                        if !sign_change_detected(g_prev[m], g_new[m], event_configs[m].direction) {
                            continue;
                        }

                        let ev = Self::find_root_brent(
                            t_a,
                            t_b,
                            &y_a,
                            &y_b,
                            &f_a,
                            &f_b,
                            g_prev[m],
                            g_new[m],
                            m,
                            &event_configs[m],
                            |ti| {
                                let yi = hermite_interp(t_a, t_b, &y_a, &y_b, &f_a, &f_b, ti);
                                events.eval(ti, &yi)[m]
                            },
                        )?;

                        let is_earlier = match &earliest {
                            None => true,
                            Some(prev) => (ev.t - t_a) * direction < (prev.t - t_a) * direction,
                        };
                        if is_earlier {
                            earliest_action = event_configs[m].action;
                            earliest = Some(ev);
                        }
                    }
                }

                if let Some(ev) = earliest {
                    match earliest_action {
                        EventAction::Stop => {
                            return Ok((IntegrationResult::Event(ev), collected));
                        }
                        EventAction::Continue => {
                            collected.push(ev);
                            t = result.t;
                            y = result.y;
                            g_prev = g_new;
                            h = result.h_next.clamp(config.h_min, config.h_max) * direction;
                            continue;
                        }
                    }
                }

                t = result.t;
                y = result.y;
                if !y.iter().all(|v| v.norm().is_finite()) {
                    return Err(IntegrationError::NonFiniteState { t });
                }
                g_prev = g_new;
            }

            h = result.h_next.clamp(config.h_min, config.h_max) * direction;

            step_count += 1;
            if step_count > config.max_steps {
                return Err(IntegrationError::MaxStepsExceeded);
            }

            if !result.accepted
                && result.h_next <= config.h_min
                && (config.tf - t) * direction > config.h_min
            {
                return Err(IntegrationError::StepSizeTooSmall {
                    t,
                    h: result.h_next,
                });
            }
        }

        Ok((IntegrationResult::Completed { t, y }, collected))
    }

    /// Find the precise root of a scalar function using Brent's method
    /// with Hermite interpolation for the state.
    #[allow(clippy::too_many_arguments)]
    fn find_root_brent(
        t_a: T::Real,
        t_b: T::Real,
        y_a: &[T; N],
        y_b: &[T; N],
        f_a: &[T; N],
        f_b: &[T; N],
        g_a: T::Real,
        g_b: T::Real,
        event_index: usize,
        config: &EventConfig<T::Real>,
        mut eval_g: impl FnMut(T::Real) -> T::Real,
    ) -> Result<EventResult<T, N>, IntegrationError<T::Real>> {
        let brent = BrentSolver::new(config.root_tol, config.max_iter);

        match brent.find_root(&mut eval_g, t_a, t_b, Some(g_a), Some(g_b)) {
            Ok((t_event, g_value, iterations)) => {
                let y_event = hermite_interp(t_a, t_b, y_a, y_b, f_a, f_b, t_event);
                Ok(EventResult {
                    t: t_event,
                    y: y_event,
                    g_value,
                    event_index,
                    iterations,
                })
            }
            Err(BrentError::NotBracketed { .. }) => Err(IntegrationError::EventFindingFailed {
                message: "Root not bracketed despite sign change detection".to_string(),
            }),
            Err(BrentError::MaxIterations {
                current_best,
                f_value,
                iterations,
            }) => {
                let y_event = hermite_interp(t_a, t_b, y_a, y_b, f_a, f_b, current_best);
                Ok(EventResult {
                    t: current_best,
                    y: y_event,
                    g_value: f_value,
                    event_index,
                    iterations,
                })
            }
        }
    }
}

/// Result of integration with event detection
#[derive(Debug, Clone)]
pub enum IntegrationResult<T: Scalar, const N: usize> {
    /// Integration completed normally (reached final time)
    Completed {
        /// Final time
        t: T::Real,
        /// Final state vector
        y: [T; N],
    },
    /// Integration stopped at an event
    Event(EventResult<T, N>),
}

/// Errors that can occur during integration
#[derive(Debug, Clone)]
pub enum IntegrationError<R: Float> {
    /// Step size became too small
    StepSizeTooSmall {
        /// Time at which step size became too small
        t: R,
        /// Step size that was too small
        h: R,
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
        t: R,
    },
}

impl<R: Float> std::fmt::Display for IntegrationError<R> {
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

impl<R: Float> std::error::Error for IntegrationError<R> {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Harmonic oscillator: y'' + w^2*y = 0
    /// State: [y, y']
    struct HarmonicOscillator {
        omega: f64,
    }

    impl OdeSystem<f64, 2> for HarmonicOscillator {
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
        // Exact solution: y = cos(wt), y' = -w*sin(wt)
        let y0 = [1.0, 0.0];
        let t0 = 0.0;
        let tf = 2.0 * std::f64::consts::PI; // One period

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);

        let (t_final, y_final) = solver
            .integrate(&sys, &IntegrationConfig::new(t0, tf, 0.1), &y0)
            .unwrap();

        // Should return to initial conditions after one period
        assert!((t_final - tf).abs() < 1e-10);
        assert!(
            (y_final[0] - 1.0).abs() < 1e-10,
            "y(2pi) = {}, expected 1.0",
            y_final[0]
        );
        assert!(
            y_final[1].abs() < 1e-10,
            "y'(2pi) = {}, expected 0.0",
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

        impl OdeSystem<f64, 1> for ExpDecay {
            fn rhs(&self, _t: f64, y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = -y[0];
            }
        }

        let sys = ExpDecay;
        let y0 = [1.0];
        let tf = 5.0;

        let tol = Tolerances::new(1e-14, 1e-14);
        let mut solver = Rkf78::new(tol);

        let (_, y_final) = solver
            .integrate(&sys, &IntegrationConfig::new(0.0, tf, 0.1), &y0)
            .unwrap();
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

    impl OdeSystem<f64, 6> for TwoBody {
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
        let mu = 398600.4418; // km^3/s^2 (Earth)
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

        let (_, y_final) = solver
            .integrate(&sys, &IntegrationConfig::new(0.0, period, 60.0), &y0)
            .unwrap();

        let e_final = compute_energy(&y_final);
        let rel_energy_error = (e_final - e0).abs() / e0.abs();

        // For RKF78 with tol=1e-12, energy drift should be very small
        assert!(
            rel_energy_error < 1e-10,
            "Energy drift {} exceeds threshold",
            rel_energy_error
        );

        println!("Two-body energy conservation test passed:");
        println!("  Initial energy: {:.15e} km^2/s^2", e0);
        println!("  Final energy:   {:.15e} km^2/s^2", e_final);
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
        impl OdeSystem<f64, 1> for CosODE {
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

    impl EventFunction<f64, 1> for ThresholdEvent {
        fn eval(&self, _t: f64, y: &[f64; 1]) -> f64 {
            y[0] - self.threshold
        }
    }

    #[test]
    fn test_event_finding_exponential() {
        // y' = y, y(0) = 1, solution: y = e^t
        // Find when y = e (should be t = 1)
        struct ExpGrowth;
        impl OdeSystem<f64, 1> for ExpGrowth {
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
        let (result, _) = solver
            .integrate_to_event(
                &sys,
                &event,
                &config,
                &IntegrationConfig::new(0.0, 10.0, 0.1),
                &y0,
            )
            .unwrap();

        match result {
            IntegrationResult::Event(ev) => {
                println!("Event found at t = {:.12}", ev.t);
                println!("  y = {:.12}", ev.y[0]);
                println!("  g = {:.3e}", ev.g_value);
                println!("  iterations: {}", ev.iterations);

                // Should find t ~ 1.0
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
        let mu = 398600.4418; // km^3/s^2 (Earth)

        struct TwoBodyForEvent {
            mu: f64,
        }

        impl OdeSystem<f64, 6> for TwoBodyForEvent {
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
        impl EventFunction<f64, 6> for RadialVelocityEvent {
            fn eval(&self, _t: f64, y: &[f64; 6]) -> f64 {
                // r_dot = (r . v) / |r|
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

        let (result, _) = solver
            .integrate_to_event(
                &sys,
                &event,
                &config,
                &IntegrationConfig::new(0.0, period, 60.0),
                &y0,
            )
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
        impl OdeSystem<f64, 1> for LinearODE {
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
        let (result, _) = solver
            .integrate_to_event(
                &LinearODE,
                &event,
                &config,
                &IntegrationConfig::new(0.0, 5.0, 0.1),
                &y0,
            )
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
        impl OdeSystem<f64, 1> for Dummy {
            fn rhs(&self, _t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = 0.0;
            }
        }
        let result = solver.integrate(&Dummy, &IntegrationConfig::new(0.0, 1.0, 0.1), &[1.0]);
        assert!(matches!(result, Err(IntegrationError::InvalidInput { .. })));
    }

    #[test]
    fn test_inf_tolerance_rejected() {
        let tol = Tolerances::new(f64::INFINITY, 1e-12);
        let mut solver = Rkf78::new(tol);
        struct Dummy;
        impl OdeSystem<f64, 1> for Dummy {
            fn rhs(&self, _t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = 0.0;
            }
        }
        let result = solver.integrate(&Dummy, &IntegrationConfig::new(0.0, 1.0, 0.1), &[1.0]);
        assert!(matches!(result, Err(IntegrationError::InvalidInput { .. })));
    }

    #[test]
    fn test_negative_tolerance_rejected() {
        let tol = Tolerances::new(-1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);
        struct Dummy;
        impl OdeSystem<f64, 1> for Dummy {
            fn rhs(&self, _t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = 0.0;
            }
        }
        let result = solver.integrate(&Dummy, &IntegrationConfig::new(0.0, 1.0, 0.1), &[1.0]);
        assert!(matches!(result, Err(IntegrationError::InvalidInput { .. })));
    }

    #[test]
    fn test_h0_sign_ignored_backward_works() {
        // With the new API, h0 sign is ignored (abs'd in IntegrationConfig::new).
        // Direction is inferred from tf - t0. Verify backward integration with positive h0 works.
        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);
        struct Dummy;
        impl OdeSystem<f64, 1> for Dummy {
            fn rhs(&self, _t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = 0.0;
            }
        }
        // Integrating backward (tf < t0) with positive h0 — should succeed
        let result = solver.integrate(&Dummy, &IntegrationConfig::new(1.0, 0.0, 0.1), &[1.0]);
        assert!(
            result.is_ok(),
            "Backward integration with positive h0 should work, got {:?}",
            result
        );
        let (t, y) = result.unwrap();
        assert!((t - 0.0).abs() < 1e-10);
        assert_eq!(y[0], 1.0);
    }

    #[test]
    fn test_nan_initial_state_rejected() {
        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);
        struct Dummy;
        impl OdeSystem<f64, 1> for Dummy {
            fn rhs(&self, _t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = 0.0;
            }
        }
        let result = solver.integrate(&Dummy, &IntegrationConfig::new(0.0, 1.0, 0.1), &[f64::NAN]);
        assert!(matches!(result, Err(IntegrationError::InvalidInput { .. })));
    }

    #[test]
    fn test_zero_length_integration() {
        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);
        struct Dummy;
        impl OdeSystem<f64, 1> for Dummy {
            fn rhs(&self, _t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = 1.0;
            }
        }
        let (t, y) = solver
            .integrate(&Dummy, &IntegrationConfig::new(5.0, 5.0, 0.1), &[42.0])
            .unwrap();
        assert_eq!(t, 5.0);
        assert_eq!(y[0], 42.0);
    }

    // ==================== Phase 2: Expanded Test Coverage ====================

    #[test]
    fn test_backward_integration() {
        // Harmonic oscillator integrated backward from 2pi to 0
        let omega = 1.0;
        let sys = HarmonicOscillator { omega };
        let tf = 2.0 * std::f64::consts::PI;

        // Start at the known final state (should be [1, 0] after one period)
        let y0 = [1.0, 0.0];

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);

        // Integrate backward: from tf to 0 (direction inferred from tf > t0)
        let (t_final, y_final) = solver
            .integrate(&sys, &IntegrationConfig::new(tf, 0.0, 0.1), &y0)
            .unwrap();

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

        let (_, y_final) = solver
            .integrate(&sys, &IntegrationConfig::new(0.0, period, 10.0), &y0)
            .unwrap();

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
        let (_, y_final) = solver
            .integrate(&sys, &IntegrationConfig::new(0.0, tf, 10.0), &y0)
            .unwrap();

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
        impl OdeSystem<f64, 1> for SingularODE {
            fn rhs(&self, _t: f64, y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = -1.0 / (y[0] * y[0] + 1e-30);
            }
        }

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);
        // Set h_min high enough that the step controller triggers StepSizeTooSmall
        // before we hit max_steps

        // y(0) = 0.001 (start very close to singularity so step size shrinks immediately)
        let result = solver.integrate(
            &SingularODE,
            &IntegrationConfig::new(0.0, 1.0, 0.0001).with_h_min(1e-4),
            &[0.001],
        );
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

        let sys = HarmonicOscillator { omega: 1.0 };
        let y0 = [1.0, 0.0];

        let result = solver.integrate(
            &sys,
            &IntegrationConfig::new(0.0, 100.0, 0.01).with_max_steps(5),
            &y0,
        );
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
        let (t_final, y_final) = solver
            .integrate(&sys, &IntegrationConfig::new(0.0, tf, 100.0), &y0)
            .unwrap();

        // Should still get the right answer
        assert!((t_final - tf).abs() < 1e-10);
        assert!(
            (y_final[0] - 1.0).abs() < 1e-9,
            "y(2pi) = {}, expected 1.0",
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
        impl OdeSystem<f64, 1> for LinearGrowth {
            fn rhs(&self, _t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = 1.0;
            }
        }

        struct ZeroCrossing;
        impl EventFunction<f64, 1> for ZeroCrossing {
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

        let (result, _) = solver
            .integrate_to_event(
                &LinearGrowth,
                &ZeroCrossing,
                &config,
                &IntegrationConfig::new(0.0, 10.0, 0.1),
                &[-0.001],
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
        impl OdeSystem<f64, 1> for LinearGrowth {
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

        let (result, _) = solver
            .integrate_to_event(
                &LinearGrowth,
                &event,
                &config,
                &IntegrationConfig::new(0.0, 5.0, 0.1),
                &[0.0],
            )
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
        // should record the event at t ~ 1 and keep going to tf = 5.
        struct LinearODE;
        impl OdeSystem<f64, 1> for LinearODE {
            fn rhs(&self, _t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = 1.0;
            }
        }

        struct ZeroCross;
        impl EventFunction<f64, 1> for ZeroCross {
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

        let (result, collected) = solver
            .integrate_to_event(
                &LinearODE,
                &ZeroCross,
                &config,
                &IntegrationConfig::new(0.0, 5.0, 0.1),
                &[-1.0],
            )
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
            collected.len(),
            1,
            "Expected 1 collected event, got {}",
            collected.len()
        );
        let ev = &collected[0];
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

        let (_, y_final) = solver
            .integrate(&sys, &IntegrationConfig::new(0.0, tf, 60.0), &y0)
            .unwrap();
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
        let (t_mid, y_mid) = solver
            .integrate(&sys, &IntegrationConfig::new(0.0, period, 0.1), &y0)
            .unwrap();

        // Backward one period
        let mut solver2 = Rkf78::new(tol);
        let (t_final, y_final) = solver2
            .integrate(&sys, &IntegrationConfig::new(t_mid, 0.0, 0.1), &y_mid)
            .unwrap();

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
        let (_, y_loose) = solver_loose
            .integrate(&sys, &IntegrationConfig::new(0.0, tf, 0.1), &y0)
            .unwrap();
        let steps_loose = solver_loose.stats.accepted_steps;

        // Run with tight uniform tolerance
        let tol_tight = Tolerances::new(1e-13, 1e-13);
        let mut solver_tight = Rkf78::new(tol_tight);
        let (_, y_tight) = solver_tight
            .integrate(&sys, &IntegrationConfig::new(0.0, tf, 0.1), &y0)
            .unwrap();
        let steps_tight = solver_tight.stats.accepted_steps;

        // Run with per-component: tight on y[0], loose on y[1]
        let tol_mixed = Tolerances::with_components([1e-13, 1e-6], [1e-13, 1e-6]);
        let mut solver_mixed = Rkf78::new(tol_mixed);
        let (_, y_mixed) = solver_mixed
            .integrate(&sys, &IntegrationConfig::new(0.0, tf, 0.1), &y0)
            .unwrap();
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
        let ctrl = StepController::<f64>::default();
        let factor = ctrl.compute_factor(0.0);
        assert_eq!(factor, ctrl.max_factor, "error=0 should give max_factor");
    }

    #[test]
    fn test_step_controller_unit_error() {
        let ctrl = StepController::<f64>::default();
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
        let ctrl = StepController::<f64>::default();
        let factor = ctrl.compute_factor(1e-20);
        assert_eq!(
            factor, ctrl.max_factor,
            "very small error should clamp to max_factor"
        );
    }

    #[test]
    fn test_step_controller_huge_error_clamped() {
        let ctrl = StepController::<f64>::default();
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
            let (_, y_final) = solver
                .integrate(&sys, &IntegrationConfig::new(0.0, tf, 0.1), &y0)
                .unwrap();
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

        let (_, y_final) = solver
            .integrate(&sys, &IntegrationConfig::new(0.0, period, 1.0), &y0)
            .unwrap();

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
        // Event: y[0] = 0 (any direction). Zeros at pi/2, 3pi/2, 5pi/2, 7pi/2 in [0, 4pi]
        let sys = HarmonicOscillator { omega: 1.0 };

        struct PositionZero;
        impl EventFunction<f64, 2> for PositionZero {
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
        let (result, collected) = solver
            .integrate_to_event(
                &sys,
                &PositionZero,
                &config,
                &IntegrationConfig::new(0.0, tf, 0.1),
                &[1.0, 0.0],
            )
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

        // cos(t) = 0 at t = pi/2, 3pi/2, 5pi/2, 7pi/2 -> 4 crossings in [0, 4pi]
        assert!(
            collected.len() >= 4,
            "Expected at least 4 zero-crossings, got {}",
            collected.len()
        );

        // Verify the first few event times are near the expected zeros
        let pi = std::f64::consts::PI;
        let expected_times = [pi / 2.0, 3.0 * pi / 2.0, 5.0 * pi / 2.0, 7.0 * pi / 2.0];
        for (i, expected_t) in expected_times.iter().enumerate() {
            if i < collected.len() {
                let actual_t = collected[i].t;
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

    // ==================== f32 Tests ====================

    #[test]
    fn test_harmonic_oscillator_f32() {
        struct HarmonicF32 {
            omega: f32,
        }
        impl OdeSystem<f32, 2> for HarmonicF32 {
            fn rhs(&self, _t: f32, y: &[f32; 2], dydt: &mut [f32; 2]) {
                dydt[0] = y[1];
                dydt[1] = -self.omega * self.omega * y[0];
            }
        }

        let sys = HarmonicF32 { omega: 1.0 };
        let tol = Tolerances::<f32, 2>::new(1e-6, 1e-6);
        let mut solver = Rkf78::<f32, 2>::new(tol);

        let y0 = [1.0_f32, 0.0];
        let tf = 2.0 * std::f32::consts::PI;
        let (t_final, y_final) = solver
            .integrate(&sys, &IntegrationConfig::new(0.0, tf, 0.1), &y0)
            .unwrap();

        assert!((t_final - tf).abs() < 1e-4, "f32 t_final: {}", t_final);
        assert!(
            (y_final[0] - 1.0).abs() < 1e-3,
            "f32 y(2pi) = {}, expected ~1.0",
            y_final[0]
        );
        assert!(
            y_final[1].abs() < 1e-3,
            "f32 y'(2pi) = {}, expected ~0.0",
            y_final[1]
        );
    }

    #[test]
    fn test_two_body_energy_f32() {
        struct TwoBodyF32 {
            mu: f32,
        }
        impl OdeSystem<f32, 6> for TwoBodyF32 {
            fn rhs(&self, _t: f32, y: &[f32; 6], dydt: &mut [f32; 6]) {
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

        let mu: f32 = 398600.44;
        let sys = TwoBodyF32 { mu };
        let r0: f32 = 6878.0;
        let v0: f32 = (mu / r0).sqrt();
        let y0 = [r0, 0.0, 0.0, 0.0, v0, 0.0];

        let period = 2.0 * std::f32::consts::PI * (r0 * r0 * r0 / mu).sqrt();

        let tol = Tolerances::<f32, 6>::new(1e-4, 1e-4);
        let mut solver = Rkf78::<f32, 6>::new(tol);

        let compute_energy = |y: &[f32; 6]| {
            let r = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
            let v2 = y[3] * y[3] + y[4] * y[4] + y[5] * y[5];
            0.5 * v2 - mu / r
        };

        let e0 = compute_energy(&y0);
        let (_, y_final) = solver
            .integrate(&sys, &IntegrationConfig::new(0.0, period, 60.0), &y0)
            .unwrap();
        let e_final = compute_energy(&y_final);

        let rel_energy_error = (e_final - e0).abs() / e0.abs();
        // f32 has ~7 digits of precision; energy drift < 1e-3 is good
        assert!(
            rel_energy_error < 1e-3,
            "f32 energy drift {} exceeds 1e-3",
            rel_energy_error
        );
    }

    // ─── StepObserver tests ──────────────────────────────────────────────

    /// Simple observer that counts accepted steps and records the last state.
    struct StepCounter {
        count: u64,
        last_t: f64,
    }

    impl StepCounter {
        fn new() -> Self {
            Self {
                count: 0,
                last_t: 0.0,
            }
        }
    }

    impl<const N: usize> StepObserver<f64, N> for StepCounter {
        fn on_step(&mut self, t: f64, _y: &[f64; N], _h: f64, _error: f64) {
            self.count += 1;
            self.last_t = t;
        }
    }

    #[test]
    fn test_step_observer_counts_steps() {
        let sys = HarmonicOscillator { omega: 1.0 };
        let y0 = [1.0, 0.0];
        let tf = 2.0 * std::f64::consts::PI;

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);
        let mut counter = StepCounter::new();

        let (t_final, _) = solver
            .integrate_with_observer(
                &sys,
                &IntegrationConfig::new(0.0, tf, 0.1),
                &y0,
                &mut counter,
            )
            .unwrap();

        // Observer should have been called at least once
        assert!(counter.count > 0, "Observer was never called");
        // Observer count should match solver stats
        assert_eq!(
            counter.count, solver.stats.accepted_steps,
            "Observer count {} != stats accepted_steps {}",
            counter.count, solver.stats.accepted_steps
        );
        // Last observed time should be at t_final
        assert!(
            (counter.last_t - t_final).abs() < 1e-10,
            "Last observed t {} != t_final {}",
            counter.last_t,
            t_final
        );
    }

    #[test]
    fn test_step_observer_noop_matches_integrate() {
        let sys = HarmonicOscillator { omega: 1.0 };
        let y0 = [1.0, 0.0];
        let tf = 2.0 * std::f64::consts::PI;
        let config = IntegrationConfig::new(0.0, tf, 0.1);

        // integrate() delegates to integrate_with_observer(&mut ())
        // Results should be identical
        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver1 = Rkf78::new(tol);
        let (t1, y1) = solver1.integrate(&sys, &config, &y0).unwrap();

        let mut solver2 = Rkf78::new(Tolerances::new(1e-12, 1e-12));
        let (t2, y2) = solver2
            .integrate_with_observer(&sys, &config, &y0, &mut ())
            .unwrap();

        assert_eq!(t1, t2);
        assert_eq!(y1, y2);
    }

    // ─── Multi-event tests ──────────────────────────────────────────────

    #[test]
    fn test_multi_event_earliest_wins() {
        // Harmonic oscillator: y = cos(t), y' = -sin(t)
        // Event 0: y[0] = 0 (cos(t) = 0 at t = pi/2)
        // Event 1: y[0] = 0.5 (cos(t) = 0.5 at t = pi/3)
        // Event 1 should fire first (pi/3 < pi/2)
        use crate::events::MultiEventFunction;

        struct TwoThresholds;

        impl MultiEventFunction<f64, 2, 2> for TwoThresholds {
            fn eval(&self, _t: f64, y: &[f64; 2]) -> [f64; 2] {
                [y[0], y[0] - 0.5] // event 0: y=0, event 1: y=0.5
            }
        }

        let sys = HarmonicOscillator { omega: 1.0 };
        let y0 = [1.0, 0.0];
        let tf = 2.0;

        // Both events are Falling (cos decreasing from 1.0)
        let configs = [
            EventConfig {
                direction: EventDirection::Falling,
                action: EventAction::Stop,
                ..Default::default()
            },
            EventConfig {
                direction: EventDirection::Falling,
                action: EventAction::Stop,
                ..Default::default()
            },
        ];

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);

        let (result, _) = solver
            .integrate_with_multi_events(
                &sys,
                &TwoThresholds,
                &configs,
                &IntegrationConfig::new(0.0, tf, 0.1),
                &y0,
            )
            .unwrap();

        match result {
            IntegrationResult::Event(ev) => {
                // Event 1 (y=0.5) should fire first at t = pi/3
                assert_eq!(ev.event_index, 1, "Event 1 should fire first");
                let expected_t = std::f64::consts::PI / 3.0;
                assert!(
                    (ev.t - expected_t).abs() < 0.01,
                    "Event time {:.6} should be near pi/3 = {:.6}",
                    ev.t,
                    expected_t
                );
            }
            IntegrationResult::Completed { .. } => {
                panic!("Should have detected an event");
            }
        }
    }

    #[test]
    fn test_multi_event_continue_collects() {
        // Use 2 events on harmonic oscillator, both Continue.
        // Event 0: y[0] crosses zero (Falling) — cos(t) = 0 at pi/2
        // Event 1: y[1] crosses zero (Falling) — -sin(t) = 0 at pi
        use crate::events::MultiEventFunction;

        struct TwoEvents;

        impl MultiEventFunction<f64, 2, 2> for TwoEvents {
            fn eval(&self, _t: f64, y: &[f64; 2]) -> [f64; 2] {
                [y[0], y[1]] // position zero, velocity zero
            }
        }

        let sys = HarmonicOscillator { omega: 1.0 };
        let y0 = [1.0, 0.0];
        let tf = 2.0 * std::f64::consts::PI;

        let configs = [
            EventConfig {
                direction: EventDirection::Any,
                action: EventAction::Continue,
                ..Default::default()
            },
            EventConfig {
                direction: EventDirection::Any,
                action: EventAction::Continue,
                ..Default::default()
            },
        ];

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);

        let (result, collected) = solver
            .integrate_with_multi_events(
                &sys,
                &TwoEvents,
                &configs,
                &IntegrationConfig::new(0.0, tf, 0.1),
                &y0,
            )
            .unwrap();

        // Should complete (all events are Continue)
        assert!(
            matches!(result, IntegrationResult::Completed { .. }),
            "Should complete, not stop at event"
        );

        // Over one full period, each event should cross zero multiple times
        assert!(
            collected.len() >= 4,
            "Expected at least 4 collected events, got {}",
            collected.len()
        );

        // Verify event_index is set correctly (should see both 0 and 1)
        let has_event_0 = collected.iter().any(|e| e.event_index == 0);
        let has_event_1 = collected.iter().any(|e| e.event_index == 1);
        assert!(has_event_0, "Should have events from event function 0");
        assert!(has_event_1, "Should have events from event function 1");
    }

    #[test]
    fn test_multi_event_single_matches_single_api() {
        // A single-event MultiEventFunction should produce the same result
        // as the single-event integrate_to_event API.
        use crate::events::MultiEventFunction;

        struct SingleWrapper;

        impl MultiEventFunction<f64, 2, 1> for SingleWrapper {
            fn eval(&self, _t: f64, y: &[f64; 2]) -> [f64; 1] {
                [y[0]] // position zero crossing
            }
        }

        struct PositionZero;
        impl EventFunction<f64, 2> for PositionZero {
            fn eval(&self, _t: f64, y: &[f64; 2]) -> f64 {
                y[0]
            }
        }

        let sys = HarmonicOscillator { omega: 1.0 };
        let y0 = [1.0, 0.0];
        let tf = 2.0;
        let event_config = EventConfig {
            direction: EventDirection::Falling,
            action: EventAction::Stop,
            ..Default::default()
        };

        // Single-event API
        let mut solver1 = Rkf78::new(Tolerances::new(1e-12, 1e-12));
        let (result1, _) = solver1
            .integrate_to_event(
                &sys,
                &PositionZero,
                &event_config,
                &IntegrationConfig::new(0.0, tf, 0.1),
                &y0,
            )
            .unwrap();

        // Multi-event API with M=1
        let mut solver2 = Rkf78::new(Tolerances::new(1e-12, 1e-12));
        let (result2, _) = solver2
            .integrate_with_multi_events(
                &sys,
                &SingleWrapper,
                &[event_config],
                &IntegrationConfig::new(0.0, tf, 0.1),
                &y0,
            )
            .unwrap();

        // Both should find the same event
        match (result1, result2) {
            (IntegrationResult::Event(ev1), IntegrationResult::Event(ev2)) => {
                assert!(
                    (ev1.t - ev2.t).abs() < 1e-10,
                    "Single ({:.10}) and multi ({:.10}) event times should match",
                    ev1.t,
                    ev2.t
                );
                assert_eq!(ev2.event_index, 0);
            }
            _ => panic!("Both should detect an event"),
        }
    }
}
