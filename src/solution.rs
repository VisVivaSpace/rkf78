//! Dense output via Hermite cubic interpolation.
//!
//! [`Solution`] stores the trajectory from an integration and supports
//! continuous evaluation at arbitrary times via Hermite cubic interpolation
//! (O(h^4) local accuracy).

use crate::events::{hermite_interp, hermite_interp_derivative};
use crate::scalar::Scalar;

/// Dense output trajectory from an ODE integration.
///
/// Stores accepted step endpoints and their derivatives, enabling
/// Hermite cubic interpolation at any time within the integration span.
///
/// Created by [`Rkf78::integrate_dense()`](crate::Rkf78::integrate_dense).
#[derive(Debug, Clone)]
pub struct Solution<T: Scalar, const N: usize> {
    /// Times at accepted step endpoints (monotonically ordered)
    times: Vec<T::Real>,
    /// States at accepted step endpoints
    states: Vec<[T; N]>,
    /// Derivatives (dy/dt) at accepted step endpoints
    derivatives: Vec<[T; N]>,
}

impl<T: Scalar, const N: usize> Solution<T, N> {
    /// Create a new empty solution with pre-allocated capacity.
    pub(crate) fn with_capacity(cap: usize) -> Self {
        Self {
            times: Vec::with_capacity(cap),
            states: Vec::with_capacity(cap),
            derivatives: Vec::with_capacity(cap),
        }
    }

    /// Push a new data point (time, state, derivative).
    pub(crate) fn push(&mut self, t: T::Real, y: [T; N], dydt: [T; N]) {
        self.times.push(t);
        self.states.push(y);
        self.derivatives.push(dydt);
    }

    /// Evaluate the interpolated state at time `t`.
    ///
    /// Returns `None` if `t` is outside the stored time range or the
    /// solution is empty.
    pub fn eval(&self, t: T::Real) -> Option<[T; N]> {
        let (i, _) = self.find_interval(t)?;
        Some(hermite_interp(
            self.times[i],
            self.times[i + 1],
            &self.states[i],
            &self.states[i + 1],
            &self.derivatives[i],
            &self.derivatives[i + 1],
            t,
        ))
    }

    /// Evaluate the interpolated derivative at time `t`.
    ///
    /// Returns `None` if `t` is outside the stored time range or the
    /// solution is empty.
    pub fn eval_derivative(&self, t: T::Real) -> Option<[T; N]> {
        let (i, _) = self.find_interval(t)?;
        Some(hermite_interp_derivative(
            self.times[i],
            self.times[i + 1],
            &self.states[i],
            &self.states[i + 1],
            &self.derivatives[i],
            &self.derivatives[i + 1],
            t,
        ))
    }

    /// The recorded times at accepted step endpoints.
    pub fn times(&self) -> &[T::Real] {
        &self.times
    }

    /// The recorded states at accepted step endpoints.
    pub fn states(&self) -> &[[T; N]] {
        &self.states
    }

    /// The recorded derivatives at accepted step endpoints.
    pub fn derivatives(&self) -> &[[T; N]] {
        &self.derivatives
    }

    /// Number of recorded data points.
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Whether the solution is empty.
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Find the interval index `i` such that `times[i] <= t <= times[i+1]`.
    /// Works for both forward (ascending) and backward (descending) time.
    /// Returns `None` if `t` is out of range or there are fewer than 2 points.
    fn find_interval(&self, t: T::Real) -> Option<(usize, T::Real)> {
        let n = self.times.len();
        if n < 2 {
            return None;
        }

        let t0 = self.times[0];
        let tf = self.times[n - 1];

        // Determine direction (forward or backward integration)
        let forward = tf > t0;

        // Check bounds
        if forward {
            if t < t0 || t > tf {
                return None;
            }
        } else if t > t0 || t < tf {
            return None;
        }

        // Binary search for the interval
        // We want the largest i such that times[i] <= t (forward)
        // or times[i] >= t (backward)
        let mut lo = 0;
        let mut hi = n - 1;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            let in_left = if forward {
                self.times[mid] <= t
            } else {
                self.times[mid] >= t
            };
            if in_left {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        Some((lo, t))
    }
}

#[cfg(test)]
mod tests {
    use crate::{IntegrationConfig, OdeSystem, Rkf78, Tolerances};

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
    fn test_dense_output_midpoints() {
        // Harmonic oscillator: y = cos(t), y' = -sin(t)
        // Cap step size at 0.5 so Hermite O(h^4) interpolation error is bounded.
        // Worst case: h^4/384 ≈ 0.5^4/384 ≈ 1.6e-4
        let sys = HarmonicOscillator { omega: 1.0 };
        let y0 = [1.0, 0.0];
        let tf = 2.0 * std::f64::consts::PI;

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);
        let config = IntegrationConfig::new(0.0, tf, 0.1).with_h_max(0.5);

        let (_, _, sol) = solver.integrate_dense(&sys, &config, &y0).unwrap();

        assert!(sol.len() >= 2, "Solution should have at least 2 points");

        // Evaluate at midpoints of each interval and compare with analytical
        let times = sol.times();
        for i in 0..times.len() - 1 {
            let t_mid = (times[i] + times[i + 1]) / 2.0;
            let y_interp = sol.eval(t_mid).unwrap();

            let y_exact = [t_mid.cos(), -t_mid.sin()];
            let pos_err = (y_interp[0] - y_exact[0]).abs();
            let vel_err = (y_interp[1] - y_exact[1]).abs();

            // Hermite cubic O(h^4): with h_max=0.5, error ≤ ~1.6e-4
            assert!(
                pos_err < 1e-3,
                "Position error {:.2e} at t={:.4} exceeds threshold",
                pos_err,
                t_mid
            );
            assert!(
                vel_err < 1e-3,
                "Velocity error {:.2e} at t={:.4} exceeds threshold",
                vel_err,
                t_mid
            );
        }
    }

    #[test]
    fn test_dense_output_derivative() {
        // Cap step size to keep Hermite derivative error (O(h^3)) bounded.
        // Worst case: ~h^3 ≈ 0.5^3 = 0.125
        let sys = HarmonicOscillator { omega: 1.0 };
        let y0 = [1.0, 0.0];
        let tf = std::f64::consts::PI;

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);
        let config = IntegrationConfig::new(0.0, tf, 0.1).with_h_max(0.5);

        let (_, _, sol) = solver.integrate_dense(&sys, &config, &y0).unwrap();

        // Check derivative at a few points
        let test_times = [0.5, 1.0, 2.0, 2.5];
        for &t in &test_times {
            if t > tf {
                continue;
            }
            let dy = sol.eval_derivative(t).unwrap();
            // For harmonic oscillator: dy/dt = [y', y''] = [-sin(t), -cos(t)]
            let dy_exact = [-t.sin(), -t.cos()];
            let err0 = (dy[0] - dy_exact[0]).abs();
            let err1 = (dy[1] - dy_exact[1]).abs();
            assert!(
                err0 < 1e-2,
                "Derivative[0] error {:.2e} at t={:.4}",
                err0,
                t
            );
            assert!(
                err1 < 1e-2,
                "Derivative[1] error {:.2e} at t={:.4}",
                err1,
                t
            );
        }
    }

    #[test]
    fn test_dense_output_endpoints_exact() {
        let sys = HarmonicOscillator { omega: 1.0 };
        let y0 = [1.0, 0.0];
        let tf = 2.0 * std::f64::consts::PI;

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);

        let (t_final, y_final, sol) = solver
            .integrate_dense(&sys, &IntegrationConfig::new(0.0, tf, 0.1), &y0)
            .unwrap();

        // Evaluating at the initial time should reproduce y0 exactly
        let y_t0 = sol.eval(0.0).unwrap();
        assert_eq!(y_t0[0], y0[0]);
        assert_eq!(y_t0[1], y0[1]);

        // Evaluating at the final stored time should reproduce y_final
        let y_tf = sol.eval(t_final).unwrap();
        assert!((y_tf[0] - y_final[0]).abs() < 1e-14);
        assert!((y_tf[1] - y_final[1]).abs() < 1e-14);
    }

    #[test]
    fn test_dense_output_out_of_range() {
        let sys = HarmonicOscillator { omega: 1.0 };
        let y0 = [1.0, 0.0];

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);

        let (_, _, sol) = solver
            .integrate_dense(&sys, &IntegrationConfig::new(0.0, 1.0, 0.1), &y0)
            .unwrap();

        assert!(sol.eval(-0.1).is_none(), "Before t0 should be None");
        assert!(sol.eval(1.1).is_none(), "After tf should be None");
    }

    #[test]
    fn test_dense_output_backward() {
        // Integrate backward: t0=2π → tf=0, verify solution.eval() works
        let sys = HarmonicOscillator { omega: 1.0 };
        let tf = 2.0 * std::f64::consts::PI;
        let y0_at_2pi = [tf.cos(), -tf.sin()]; // [1.0, ~0.0]

        let tol = Tolerances::new(1e-12, 1e-12);
        let mut solver = Rkf78::new(tol);
        let config = IntegrationConfig::new(tf, 0.0, 0.1).with_h_max(0.5);

        let (t_final, _, sol) = solver.integrate_dense(&sys, &config, &y0_at_2pi).unwrap();
        assert!((t_final - 0.0).abs() < 1e-10, "Should integrate to t=0");
        assert!(sol.len() >= 2, "Solution needs at least 2 points");

        // Evaluate at several interior points
        let test_times = [5.0, 4.0, 3.0, 2.0, 1.0, 0.5];
        for &t in &test_times {
            let y = sol.eval(t).expect("eval should succeed for interior t");
            let y_exact = [t.cos(), -t.sin()];
            let err = (y[0] - y_exact[0]).abs();
            assert!(
                err < 1e-3,
                "Backward dense eval at t={}: error {:.2e} exceeds threshold",
                t,
                err
            );
        }

        // Boundary: out of range should be None
        assert!(sol.eval(tf + 0.1).is_none(), "Beyond t0 should be None");
        assert!(sol.eval(-0.1).is_none(), "Before tf should be None");
    }

    #[test]
    fn test_dense_output_convergence_order() {
        // Verify O(h^4) convergence of Hermite cubic interpolation.
        // Use loose integrator tolerance so the adaptive controller actually
        // hits h_max (tight tol makes steps much smaller than h_max).
        let sys = HarmonicOscillator { omega: 1.0 };
        let y0 = [1.0, 0.0];
        let tf = 1.0;
        let t_eval = 0.5;

        let mut errors = Vec::new();
        let h_maxes = [0.5, 0.25];

        for &h_max in &h_maxes {
            // Loose integrator tolerance so step control wants large steps
            let tol = Tolerances::new(1e-4, 1e-4);
            let mut solver = Rkf78::new(tol);
            let config = IntegrationConfig::new(0.0, tf, h_max).with_h_max(h_max);

            let (_, _, sol) = solver.integrate_dense(&sys, &config, &y0).unwrap();

            let y_interp = sol.eval(t_eval).unwrap();
            let y_exact = t_eval.cos();
            errors.push((y_interp[0] - y_exact).abs());
        }

        // With h halved, Hermite cubic error should decrease by ~2^4 = 16
        // Allow some margin: require at least 2^2.5 ≈ 5.6x improvement
        if errors[0] > 1e-15 && errors[1] > 1e-15 {
            let ratio = errors[0] / errors[1];
            assert!(
                ratio > 5.0,
                "Convergence ratio {:.1} should be > 5 (expected ~16 for O(h^4)), \
                 errors: {:.2e}, {:.2e}",
                ratio,
                errors[0],
                errors[1]
            );
        }
    }
}
