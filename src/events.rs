//! Event Finding for ODE Integration
//!
//! This module provides event detection during ODE integration, allowing
//! the integrator to stop precisely when a user-defined condition is met.
//!
//! # Overview
//!
//! An event function `g(t, y)` is monitored during integration. When `g`
//! changes sign (crosses zero), the integrator uses Brent's method to
//! find the precise time of the zero crossing.
//!
//! # Common Applications in Astrodynamics
//!
//! - Periapsis/apoapsis detection (radial velocity = 0)
//! - Sphere of influence crossing
//! - Eclipse entry/exit
//! - Ground track crossing (ascending/descending node)
//! - Altitude threshold crossing
//! - Conjunction/opposition detection

use crate::scalar::{Float, Scalar};

/// Event function trait
///
/// Implement this trait to define conditions that should stop the integration.
///
/// # Example
///
/// ```ignore
/// // Detect when altitude drops below 100 km
/// struct AltitudeEvent {
///     threshold: f64,
///     earth_radius: f64,
/// }
///
/// impl EventFunction<f64, 6> for AltitudeEvent {
///     fn eval(&self, _t: f64, y: &[f64; 6]) -> f64 {
///         let r = (y[0]*y[0] + y[1]*y[1] + y[2]*y[2]).sqrt();
///         let altitude = r - self.earth_radius;
///         altitude - self.threshold  // Zero when altitude = threshold
///     }
/// }
/// ```
pub trait EventFunction<T: Scalar, const N: usize> {
    /// Evaluate the event function.
    ///
    /// The integrator will stop when this function crosses zero.
    /// The direction of crossing can be specified via `EventDirection`.
    ///
    /// # Arguments
    /// * `t` - Current time
    /// * `y` - Current state vector
    ///
    /// # Returns
    /// The value of the event function (always real). Zero indicates the event has occurred.
    fn eval(&self, t: T::Real, y: &[T; N]) -> T::Real;
}

/// Direction of zero-crossing to detect
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EventDirection {
    /// Detect when g goes from negative to positive (increasing through zero)
    Rising,
    /// Detect when g goes from positive to negative (decreasing through zero)
    Falling,
    /// Detect any zero crossing
    #[default]
    Any,
}

/// Action to take when an event is detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EventAction {
    /// Stop integration at this event
    #[default]
    Stop,
    /// Record the event but continue integration
    Continue,
}

/// Configuration for an event
#[derive(Debug, Clone, Copy)]
pub struct EventConfig<R: Float> {
    /// Which direction of zero-crossing to detect
    pub direction: EventDirection,
    /// What to do when the event is detected
    pub action: EventAction,
    /// Tolerance for root finding (default: 1e-12)
    pub root_tol: R,
    /// Maximum iterations for root finding (default: 50)
    pub max_iter: usize,
}

impl<R: Float> Default for EventConfig<R> {
    fn default() -> Self {
        Self {
            direction: EventDirection::Any,
            action: EventAction::Stop,
            root_tol: R::from_f64(1e-12),
            max_iter: 50,
        }
    }
}

/// Result of event detection
#[derive(Debug, Clone, Copy)]
pub struct EventResult<T: Scalar, const N: usize> {
    /// Time at which the event occurred
    pub t: T::Real,
    /// State at the event
    pub y: [T; N],
    /// Value of the event function at the event (should be ~0)
    pub g_value: T::Real,
    /// Index of the event that fired (0 for single-event methods)
    pub event_index: usize,
    /// Number of root-finding iterations used
    pub iterations: usize,
}

/// Multiple simultaneous event functions.
///
/// Implement this trait to monitor M event functions simultaneously.
/// The integrator will detect the earliest zero crossing across all M events.
///
/// # Type Parameters
/// * `T` - Scalar type for state components
/// * `N` - Dimension of the state vector
/// * `M` - Number of event functions
pub trait MultiEventFunction<T: Scalar, const N: usize, const M: usize> {
    /// Evaluate all M event functions at once.
    ///
    /// # Arguments
    /// * `t` - Current time
    /// * `y` - Current state vector
    ///
    /// # Returns
    /// Array of M event function values (all real).
    fn eval(&self, t: T::Real, y: &[T; N]) -> [T::Real; M];
}

/// Brent's method for root finding
///
/// A robust root-finding algorithm combining bisection, secant method,
/// and inverse quadratic interpolation.
///
/// Reference: Brent, R.P. (1973). "Algorithms for Minimization without
/// Derivatives". Prentice-Hall.
pub(crate) struct BrentSolver<R: Float> {
    /// Tolerance for convergence
    pub tol: R,
    /// Maximum iterations
    pub max_iter: usize,
}

impl<R: Float> Default for BrentSolver<R> {
    fn default() -> Self {
        Self {
            tol: R::from_f64(1e-12),
            max_iter: 50,
        }
    }
}

impl<R: Float> BrentSolver<R> {
    /// Create a new Brent solver with specified tolerance
    pub fn new(tol: R, max_iter: usize) -> Self {
        Self { tol, max_iter }
    }

    /// Find the root of f in the interval [a, b].
    ///
    /// Assumes f(a) and f(b) have opposite signs (i.e., the root is bracketed).
    ///
    /// # Arguments
    /// * `f` - Function to find root of
    /// * `a` - Left endpoint of bracket
    /// * `b` - Right endpoint of bracket
    /// * `fa` - f(a) (optional, will be computed if None)
    /// * `fb` - f(b) (optional, will be computed if None)
    ///
    /// # Returns
    /// * `Ok((root, f_root, iterations))` - The root, function value at root, and iteration count
    /// * `Err(BrentError)` - If root finding fails
    pub fn find_root<F>(
        &self,
        mut f: F,
        mut a: R,
        mut b: R,
        fa: Option<R>,
        fb: Option<R>,
    ) -> Result<(R, R, usize), BrentError<R>>
    where
        F: FnMut(R) -> R,
    {
        let mut fa = fa.unwrap_or_else(|| f(a));
        let mut fb = fb.unwrap_or_else(|| f(b));

        // Check that root is bracketed
        if fa * fb > R::ZERO {
            return Err(BrentError::NotBracketed { a, b, fa, fb });
        }

        // Ensure |f(a)| >= |f(b)|
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }

        let mut c = a;
        let mut fc = fa;
        let mut mflag = true;
        let mut d = b - a; // previous step size

        let three = R::from_f64(3.0);
        let four = R::from_f64(4.0);

        for iter in 0..self.max_iter {
            // Ensure |f(a)| >= |f(b)| so b is the best guess
            if fa.abs() < fb.abs() {
                std::mem::swap(&mut a, &mut b);
                std::mem::swap(&mut fa, &mut fb);
            }

            // Check for convergence
            if fb == R::ZERO || (b - a).abs() <= self.tol {
                return Ok((b, fb, iter + 1));
            }

            // Try inverse quadratic interpolation or secant
            let s = if fa != fc && fb != fc && fa != fb {
                // Inverse quadratic interpolation
                a * fb * fc / ((fa - fb) * (fa - fc))
                    + b * fa * fc / ((fb - fa) * (fb - fc))
                    + c * fa * fb / ((fc - fa) * (fc - fb))
            } else if fb != fa {
                // Secant method
                b - fb * (b - a) / (fb - fa)
            } else {
                // Degenerate: fa == fb, fall back to bisection
                (a + b) / R::TWO
            };

            // Conditions for rejecting s and falling back to bisection
            let mid = (a + b) / R::TWO;
            let use_bisection =
                // s not between (3a+b)/4 and b
                (s - (three * a + b) / four) * (s - b) > R::ZERO
                // |s-b| >= |b-c|/2 when mflag set (last step was bisection)
                || (mflag && (s - b).abs() >= (b - c).abs() / R::TWO)
                // |s-b| >= |c-d|/2 when mflag not set
                || (!mflag && (s - b).abs() >= (c - d).abs() / R::TWO)
                // |b-c| < tol when mflag set
                || (mflag && (b - c).abs() < self.tol)
                // |c-d| < tol when mflag not set
                || (!mflag && (c - d).abs() < self.tol);

            let s = if use_bisection {
                mflag = true;
                mid
            } else {
                mflag = false;
                s
            };

            let fs = f(s);
            d = c; // d = previous c (two steps back)
            c = b;
            fc = fb;

            if fa * fs < R::ZERO {
                b = s;
                fb = fs;
            } else {
                a = s;
                fa = fs;
            }
        }

        Err(BrentError::MaxIterations {
            current_best: b,
            f_value: fb,
            iterations: self.max_iter,
        })
    }
}

/// Errors from Brent's method
#[derive(Debug, Clone)]
pub(crate) enum BrentError<R: Float> {
    /// The root is not bracketed by the given interval
    NotBracketed {
        /// Left endpoint
        a: R,
        /// Right endpoint
        b: R,
        /// Function value at left endpoint
        fa: R,
        /// Function value at right endpoint
        fb: R,
    },
    /// Maximum iterations reached without convergence
    MaxIterations {
        /// Best root estimate so far
        current_best: R,
        /// Function value at best estimate
        f_value: R,
        /// Number of iterations performed
        iterations: usize,
    },
}

impl<R: Float> std::fmt::Display for BrentError<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BrentError::NotBracketed { a, b, fa, fb } => {
                write!(
                    f,
                    "Root not bracketed: f({}) = {}, f({}) = {} (same sign)",
                    a, fa, b, fb
                )
            }
            BrentError::MaxIterations {
                current_best,
                f_value,
                iterations,
            } => {
                write!(
                    f,
                    "Max iterations ({}) reached, best estimate: {}, f = {}",
                    iterations, current_best, f_value
                )
            }
        }
    }
}

impl<R: Float> std::error::Error for BrentError<R> {}

/// Check if a sign change occurred in the specified direction
pub(crate) fn sign_change_detected<R: Float>(
    g_old: R,
    g_new: R,
    direction: EventDirection,
) -> bool {
    if g_old * g_new > R::ZERO {
        // No sign change
        return false;
    }

    if g_new == R::ZERO {
        // New value exactly at zero - consider this a detection
        return true;
    }

    if g_old == R::ZERO {
        // Old value exactly at zero - not a new crossing, skip it
        return false;
    }

    match direction {
        EventDirection::Rising => g_old < R::ZERO && g_new > R::ZERO,
        EventDirection::Falling => g_old > R::ZERO && g_new < R::ZERO,
        EventDirection::Any => true,
    }
}

/// Hermite cubic interpolation between two endpoints.
///
/// Given state and derivative at `t_a` and `t_b`, compute the interpolated
/// state at time `t` with O(h^4) local accuracy.
///
/// # Arguments
/// * `t_a`, `t_b` — Endpoint times
/// * `y_a`, `y_b` — States at endpoints
/// * `f_a`, `f_b` — Derivatives (dy/dt) at endpoints
/// * `t` — Interpolation time (should be in `[t_a, t_b]`)
pub(crate) fn hermite_interp<T: Scalar, const N: usize>(
    t_a: T::Real,
    t_b: T::Real,
    y_a: &[T; N],
    y_b: &[T; N],
    f_a: &[T; N],
    f_b: &[T; N],
    t: T::Real,
) -> [T; N] {
    let dt = t_b - t_a;
    let alpha = (t - t_a) / dt;
    let a2 = alpha * alpha;
    let a3 = a2 * alpha;

    let one = T::Real::ONE;
    let two = T::Real::TWO;
    let three = T::Real::from_f64(3.0);

    // Hermite basis functions
    let h00 = one - three * a2 + two * a3; // y_a weight
    let h10 = alpha - two * a2 + a3; // f_a weight (scaled by dt)
    let h01 = three * a2 - two * a3; // y_b weight
    let h11 = -a2 + a3; // f_b weight (scaled by dt)

    let mut y = [T::ZERO; N];
    for i in 0..N {
        y[i] = y_a[i].mul_real(h00)
            + f_a[i].mul_real(h10 * dt)
            + y_b[i].mul_real(h01)
            + f_b[i].mul_real(h11 * dt);
    }
    y
}

/// Hermite cubic derivative interpolation between two endpoints.
///
/// Given state and derivative at `t_a` and `t_b`, compute the interpolated
/// derivative at time `t`.
pub(crate) fn hermite_interp_derivative<T: Scalar, const N: usize>(
    t_a: T::Real,
    t_b: T::Real,
    y_a: &[T; N],
    y_b: &[T; N],
    f_a: &[T; N],
    f_b: &[T; N],
    t: T::Real,
) -> [T; N] {
    let dt = t_b - t_a;
    let alpha = (t - t_a) / dt;
    let a2 = alpha * alpha;

    let two = T::Real::TWO;
    let three = T::Real::from_f64(3.0);
    let six = T::Real::from_f64(6.0);

    // Derivatives of Hermite basis functions w.r.t. alpha, divided by dt
    // d(h00)/dt = (-6α + 6α²) / dt
    // d(h10)/dt = (1 - 4α + 3α²) / dt
    // d(h01)/dt = (6α - 6α²) / dt
    // d(h11)/dt = (-2α + 3α²) / dt
    let dh00 = (-six * alpha + six * a2) / dt;
    let dh10 = (T::Real::ONE - two * two * alpha + three * a2) / dt;
    let dh01 = (six * alpha - six * a2) / dt;
    let dh11 = (-two * alpha + three * a2) / dt;

    let mut dy = [T::ZERO; N];
    for i in 0..N {
        dy[i] = y_a[i].mul_real(dh00)
            + f_a[i].mul_real(dh10 * dt)
            + y_b[i].mul_real(dh01)
            + f_b[i].mul_real(dh11 * dt);
    }
    dy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brent_simple_root() {
        let solver = BrentSolver::<f64>::default();

        // f(x) = x^2 - 2, root at sqrt(2) ≈ 1.414
        let result = solver.find_root(|x| x * x - 2.0, 0.0, 2.0, None, None);

        let (root, f_root, iters) = result.unwrap();
        let expected = 2.0_f64.sqrt();

        assert!(
            (root - expected).abs() < 1e-13,
            "Root {} should be close to sqrt(2) = {}",
            root,
            expected
        );
        assert!(f_root.abs() < 1e-13, "f(root) = {} should be ~0", f_root);
        println!(
            "Found root {} in {} iterations (exact: {})",
            root, iters, expected
        );
    }

    #[test]
    fn test_brent_trigonometric() {
        let solver = BrentSolver::<f64>::default();

        // f(x) = sin(x), root at π
        let result = solver.find_root(|x| x.sin(), 3.0, 4.0, None, None);

        let (root, f_root, iters) = result.unwrap();
        let expected = std::f64::consts::PI;

        assert!(
            (root - expected).abs() < 1e-13,
            "Root {} should be close to π = {}",
            root,
            expected
        );
        assert!(f_root.abs() < 1e-13);
        println!("Found root {} in {} iterations (exact: π)", root, iters);
    }

    #[test]
    fn test_brent_not_bracketed() {
        let solver = BrentSolver::<f64>::default();

        // f(x) = x^2 + 1, no real roots
        let result = solver.find_root(|x| x * x + 1.0, -1.0, 1.0, None, None);

        assert!(matches!(result, Err(BrentError::NotBracketed { .. })));
    }

    #[test]
    fn test_sign_change_detection() {
        // Rising edge
        assert!(sign_change_detected(-1.0_f64, 1.0, EventDirection::Rising));
        assert!(!sign_change_detected(1.0_f64, -1.0, EventDirection::Rising));
        assert!(sign_change_detected(-1.0_f64, 1.0, EventDirection::Any));

        // Falling edge
        assert!(sign_change_detected(1.0_f64, -1.0, EventDirection::Falling));
        assert!(!sign_change_detected(
            -1.0_f64,
            1.0,
            EventDirection::Falling
        ));
        assert!(sign_change_detected(1.0_f64, -1.0, EventDirection::Any));

        // No sign change
        assert!(!sign_change_detected(1.0_f64, 2.0, EventDirection::Any));
        assert!(!sign_change_detected(-1.0_f64, -2.0, EventDirection::Any));
    }

    #[test]
    fn test_brent_root_at_endpoint() {
        // f(x) = x + 1, root at x = -1 (left bracket endpoint)
        let solver = BrentSolver::<f64>::default();
        let result = solver.find_root(|x| x + 1.0, -1.0, 1.0, None, None);
        let (root, f_root, _) = result.unwrap();
        assert!(
            (root - (-1.0)).abs() < 1e-12,
            "Root {} should be -1.0",
            root
        );
        assert!(f_root.abs() < 1e-12);
    }

    #[test]
    fn test_brent_triple_root() {
        // f(x) = (x-1)^3, triple root at x = 1, bracket [0, 2]
        // Triple roots are hard for Brent because convergence degrades.
        // We accept finding the root within a looser tolerance.
        let solver = BrentSolver::<f64>::new(1e-12, 100);
        let result = solver.find_root(|x| (x - 1.0_f64).powi(3), 0.0, 2.0, None, None);
        let (root, _, _) = result.unwrap();
        assert!(
            (root - 1.0).abs() < 1e-4,
            "Triple root {} should be near 1.0",
            root
        );
    }

    #[test]
    fn test_brent_near_zero_bracket() {
        // f(x) = x, root at 0, bracket [-1e-15, 1e-15]
        // The bracket is smaller than the default tol (1e-12), so Brent
        // converges immediately and returns the best endpoint.
        let solver = BrentSolver::<f64>::default();
        let result = solver.find_root(|x| x, -1e-15, 1e-15, None, None);
        let (root, _, _) = result.unwrap();
        // Root must be within the original bracket
        assert!(
            root.abs() <= 1e-15,
            "Root {} should be within bracket [-1e-15, 1e-15]",
            root
        );
    }

    #[test]
    fn test_brent_equal_function_values() {
        // f(x) = (x - 0.5)^3: f(0) = -0.125, f(1) = 0.125
        // Symmetric about the root — early iterations may produce fa == fb.
        // This exercises the degenerate bisection fallback.
        let solver = BrentSolver::<f64>::default();
        let result = solver.find_root(|x| (x - 0.5_f64).powi(3), 0.0, 1.0, None, None);
        let (root, _, _) = result.unwrap();
        assert!(
            (root - 0.5).abs() < 1e-4,
            "Root {} should be near 0.5",
            root
        );
    }

    #[test]
    fn test_brent_cubic() {
        let solver = BrentSolver::<f64>::default();

        // f(x) = x^3 - x - 2, has a root near 1.52
        let result = solver.find_root(|x| x.powi(3) - x - 2.0, 1.0, 2.0, None, None);

        let (root, f_root, iters) = result.unwrap();

        // Verify it's actually a root
        assert!(f_root.abs() < 1e-12);
        // Verify the root value
        assert!((root - 1.5213797068045676).abs() < 1e-10);
        println!("Cubic root found: {} in {} iterations", root, iters);
    }
}
