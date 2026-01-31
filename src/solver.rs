//! Runge-Kutta-Fehlberg 7(8) Integrator
//!
//! A 13-stage embedded RK7(8) pair for high-precision integration of ODEs.
//! Designed for spacecraft trajectory propagation and astrodynamics applications.
//!
//! Reference: NASA TR R-287, Erwin Fehlberg, 1968

use crate::coefficients::{A, B, B_ERR, C, STAGES};
use crate::events::{
    sign_change_detected, BrentError, BrentSolver, EventConfig, EventFunction, EventResult,
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
        let h = h.clamp(self.h_min, self.h_max);
        
        // Compute all 13 stages
        self.compute_stages(sys, t, y, h);
        
        // Compute 8th order solution
        let y8 = self.compute_solution(y, h);
        
        // Compute error estimate
        let error = self.compute_error(&y8, h);
        
        // Determine acceptance
        let accepted = error <= 1.0;
        
        // Compute next step size
        let factor = self.controller.compute_factor(error);
        let h_next = (h * factor).clamp(self.h_min, self.h_max);
        
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
            
            // Check for step size too small
            if h.abs() < self.h_min && (tf - t) * direction > self.h_min {
                return Err(IntegrationError::StepSizeTooSmall { 
                    t, 
                    h: h.abs() 
                });
            }
        }
        
        Ok((t, y))
    }
    
    /// Compute all 13 stages
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
                        sys,
                        event,
                        t,
                        &y,
                        result.t,
                        &result.y,
                        g_prev,
                        g_new,
                        config,
                    )?;

                    return Ok(IntegrationResult::Event(event_result));
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

            if h.abs() < self.h_min && (tf - t) * direction > self.h_min {
                return Err(IntegrationError::StepSizeTooSmall { t, h: h.abs() });
            }
        }

        // Reached tf without event
        Ok(IntegrationResult::Completed { t, y })
    }

    /// Find the precise root location using Brent's method.
    ///
    /// This interpolates the state between two integration steps to
    /// evaluate the event function at intermediate times.
    fn find_event_root<S, E>(
        &mut self,
        _sys: &S,
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

        // We need to be able to evaluate the state at any time t in [t_a, t_b].
        // The simplest approach is to re-integrate from t_a with small steps.
        // A more sophisticated approach would use dense output / interpolation.
        //
        // For now, we use a simple linear interpolation of the state,
        // which is adequate when the event occurs close to t_a or t_b.
        // For higher accuracy, implement dense output using the RK stages.

        let dt = t_b - t_a;

        // Create a function that evaluates g at time t by interpolating the state
        let eval_g = |t: f64| {
            // Linear interpolation factor
            let alpha = (t - t_a) / dt;

            // Interpolate state (simple linear interpolation)
            let mut y_interp = [0.0; N];
            for i in 0..N {
                y_interp[i] = y_a[i] + alpha * (y_b[i] - y_a[i]);
            }

            event.eval(t, &y_interp)
        };

        // Find the root
        match solver.find_root(eval_g, t_a, t_b, Some(g_a), Some(g_b)) {
            Ok((t_event, g_value, iterations)) => {
                // Compute the state at the event time
                // For higher precision, re-integrate to t_event
                let alpha = (t_event - t_a) / dt;
                let mut y_event = [0.0; N];
                for i in 0..N {
                    y_event[i] = y_a[i] + alpha * (y_b[i] - y_a[i]);
                }

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
                let alpha = (current_best - t_a) / dt;
                let mut y_event = [0.0; N];
                for i in 0..N {
                    y_event[i] = y_a[i] + alpha * (y_b[i] - y_a[i]);
                }

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
        assert!((y_final[0] - 1.0).abs() < 1e-10, "y(2π) = {}, expected 1.0", y_final[0]);
        assert!(y_final[1].abs() < 1e-10, "y'(2π) = {}, expected 0.0", y_final[1]);
        
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
        mu: f64,  // GM parameter
    }
    
    impl OdeSystem<6> for TwoBody {
        fn rhs(&self, _t: f64, y: &[f64; 6], dydt: &mut [f64; 6]) {
            let x = y[0];
            let y_pos = y[1];
            let z = y[2];
            
            let r = (x*x + y_pos*y_pos + z*z).sqrt();
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
            let r = (y[0]*y[0] + y[1]*y[1] + y[2]*y[2]).sqrt();
            let v2 = y[3]*y[3] + y[4]*y[4] + y[5]*y[5];
            0.5 * v2 - mu / r
        };
        
        let e0 = compute_energy(&y0);
        
        let (_, y_final) = solver.integrate(&sys, 0.0, &y0, period, 60.0).unwrap();
        
        let e_final = compute_energy(&y_final);
        let rel_energy_error = (e_final - e0).abs() / e0.abs();
        
        // For RKF78 with tol=1e-12, energy drift should be very small
        assert!(rel_energy_error < 1e-10, 
            "Energy drift {} exceeds threshold", rel_energy_error);
        
        println!("Two-body energy conservation test passed:");
        println!("  Initial energy: {:.15e} km²/s²", e0);
        println!("  Final energy:   {:.15e} km²/s²", e_final);
        println!("  Relative drift: {:.3e}", rel_energy_error);
        println!("  Stats: {:?}", solver.stats);
    }
    
    #[test]
    fn test_order_of_convergence() {
        // Test that halving step size reduces error by ~2^8 = 256
        // Use a problem with known solution
        
        struct SinusoidalODE;
        impl OdeSystem<1> for SinusoidalODE {
            fn rhs(&self, t: f64, _y: &[f64; 1], dydt: &mut [f64; 1]) {
                dydt[0] = t.cos();  // y' = cos(t), y = sin(t) + C
            }
        }
        
        let sys = SinusoidalODE;
        let y0 = [0.0];  // y(0) = 0, so y = sin(t)
        let tf = 1.0;
        
        // Very loose tolerance to force fixed-ish step behavior
        let tol = Tolerances::new(1e-4, 1e-4);
        
        // This is more of a sanity check than a rigorous order test
        // A full order test would need to control step size more carefully
        let mut solver = Rkf78::new(tol);
        let (_, y_final) = solver.integrate(&sys, 0.0, &y0, tf, 0.1).unwrap();
        
        let exact = tf.sin();
        let error = (y_final[0] - exact).abs();
        
        println!("Order convergence test:");
        println!("  y({}) = {:.15}, exact = {:.15}", tf, y_final[0], exact);
        println!("  Error: {:.3e}", error);
        
        // With these tolerances, error should be quite small
        assert!(error < 1e-4);
    }

    // ==================== Event Finding Tests ====================

    use crate::events::{EventConfig, EventDirection, EventFunction};

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
}
