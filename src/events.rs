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
/// impl EventFunction<6> for AltitudeEvent {
///     fn eval(&self, _t: f64, y: &[f64; 6]) -> f64 {
///         let r = (y[0]*y[0] + y[1]*y[1] + y[2]*y[2]).sqrt();
///         let altitude = r - self.earth_radius;
///         altitude - self.threshold  // Zero when altitude = threshold
///     }
/// }
/// ```
pub trait EventFunction<const N: usize> {
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
    /// The value of the event function. Zero indicates the event has occurred.
    fn eval(&self, t: f64, y: &[f64; N]) -> f64;
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
#[derive(Debug, Clone)]
pub struct EventConfig {
    /// Which direction of zero-crossing to detect
    pub direction: EventDirection,
    /// What to do when the event is detected
    pub action: EventAction,
    /// Tolerance for root finding (default: 1e-12)
    pub root_tol: f64,
    /// Maximum iterations for root finding (default: 50)
    pub max_iter: usize,
}

impl Default for EventConfig {
    fn default() -> Self {
        Self {
            direction: EventDirection::Any,
            action: EventAction::Stop,
            root_tol: 1e-12,
            max_iter: 50,
        }
    }
}

/// Result of event detection
#[derive(Debug, Clone)]
pub struct EventResult<const N: usize> {
    /// Time at which the event occurred
    pub t: f64,
    /// State at the event
    pub y: [f64; N],
    /// Value of the event function at the event (should be ~0)
    pub g_value: f64,
    /// Number of root-finding iterations used
    pub iterations: usize,
}

/// Brent's method for root finding
///
/// A robust root-finding algorithm combining bisection, secant method,
/// and inverse quadratic interpolation.
///
/// Reference: Brent, R.P. (1973). "Algorithms for Minimization without
/// Derivatives". Prentice-Hall.
pub struct BrentSolver {
    /// Tolerance for convergence
    pub tol: f64,
    /// Maximum iterations
    pub max_iter: usize,
}

impl Default for BrentSolver {
    fn default() -> Self {
        Self {
            tol: 1e-12,
            max_iter: 50,
        }
    }
}

impl BrentSolver {
    /// Create a new Brent solver with specified tolerance
    pub fn new(tol: f64, max_iter: usize) -> Self {
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
        mut a: f64,
        mut b: f64,
        fa: Option<f64>,
        fb: Option<f64>,
    ) -> Result<(f64, f64, usize), BrentError>
    where
        F: FnMut(f64) -> f64,
    {
        let mut fa = fa.unwrap_or_else(|| f(a));
        let mut fb = fb.unwrap_or_else(|| f(b));

        // Check that root is bracketed
        if fa * fb > 0.0 {
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

        for iter in 0..self.max_iter {
            // Ensure |f(a)| >= |f(b)| so b is the best guess
            if fa.abs() < fb.abs() {
                std::mem::swap(&mut a, &mut b);
                std::mem::swap(&mut fa, &mut fb);
            }

            // Check for convergence
            if fb == 0.0 || (b - a).abs() <= self.tol {
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
                (a + b) / 2.0
            };

            // Conditions for rejecting s and falling back to bisection
            let mid = (a + b) / 2.0;
            let use_bisection =
                // s not between (3a+b)/4 and b
                (s - (3.0 * a + b) / 4.0) * (s - b) > 0.0
                // |s-b| >= |b-c|/2 when mflag set (last step was bisection)
                || (mflag && (s - b).abs() >= (b - c).abs() / 2.0)
                // |s-b| >= |c-d|/2 when mflag not set
                || (!mflag && (s - b).abs() >= (c - d).abs() / 2.0)
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

            if fa * fs < 0.0 {
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
pub enum BrentError {
    /// The root is not bracketed by the given interval
    NotBracketed {
        /// Left endpoint
        a: f64,
        /// Right endpoint
        b: f64,
        /// Function value at left endpoint
        fa: f64,
        /// Function value at right endpoint
        fb: f64,
    },
    /// Maximum iterations reached without convergence
    MaxIterations {
        /// Best root estimate so far
        current_best: f64,
        /// Function value at best estimate
        f_value: f64,
        /// Number of iterations performed
        iterations: usize,
    },
}

impl std::fmt::Display for BrentError {
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

impl std::error::Error for BrentError {}

/// Check if a sign change occurred in the specified direction
pub fn sign_change_detected(g_old: f64, g_new: f64, direction: EventDirection) -> bool {
    if g_old * g_new > 0.0 {
        // No sign change
        return false;
    }

    if g_new == 0.0 {
        // New value exactly at zero - consider this a detection
        return true;
    }

    if g_old == 0.0 {
        // Old value exactly at zero - not a new crossing, skip it
        return false;
    }

    match direction {
        EventDirection::Rising => g_old < 0.0 && g_new > 0.0,
        EventDirection::Falling => g_old > 0.0 && g_new < 0.0,
        EventDirection::Any => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brent_simple_root() {
        let solver = BrentSolver::default();

        // f(x) = x^2 - 2, root at sqrt(2) ≈ 1.414
        let result = solver.find_root(|x| x * x - 2.0, 0.0, 2.0, None, None);

        let (root, f_root, iters) = result.unwrap();
        let expected = 2.0_f64.sqrt();

        assert!(
            (root - expected).abs() < 1e-12,
            "Root {} should be close to sqrt(2) = {}",
            root,
            expected
        );
        assert!(f_root.abs() < 1e-12, "f(root) = {} should be ~0", f_root);
        println!(
            "Found root {} in {} iterations (exact: {})",
            root, iters, expected
        );
    }

    #[test]
    fn test_brent_trigonometric() {
        let solver = BrentSolver::default();

        // f(x) = sin(x), root at π
        let result = solver.find_root(|x| x.sin(), 3.0, 4.0, None, None);

        let (root, f_root, iters) = result.unwrap();
        let expected = std::f64::consts::PI;

        assert!(
            (root - expected).abs() < 1e-12,
            "Root {} should be close to π = {}",
            root,
            expected
        );
        assert!(f_root.abs() < 1e-12);
        println!("Found root {} in {} iterations (exact: π)", root, iters);
    }

    #[test]
    fn test_brent_not_bracketed() {
        let solver = BrentSolver::default();

        // f(x) = x^2 + 1, no real roots
        let result = solver.find_root(|x| x * x + 1.0, -1.0, 1.0, None, None);

        assert!(matches!(result, Err(BrentError::NotBracketed { .. })));
    }

    #[test]
    fn test_sign_change_detection() {
        // Rising edge
        assert!(sign_change_detected(-1.0, 1.0, EventDirection::Rising));
        assert!(!sign_change_detected(1.0, -1.0, EventDirection::Rising));
        assert!(sign_change_detected(-1.0, 1.0, EventDirection::Any));

        // Falling edge
        assert!(sign_change_detected(1.0, -1.0, EventDirection::Falling));
        assert!(!sign_change_detected(-1.0, 1.0, EventDirection::Falling));
        assert!(sign_change_detected(1.0, -1.0, EventDirection::Any));

        // No sign change
        assert!(!sign_change_detected(1.0, 2.0, EventDirection::Any));
        assert!(!sign_change_detected(-1.0, -2.0, EventDirection::Any));
    }

    #[test]
    fn test_brent_root_at_endpoint() {
        // f(x) = x + 1, root at x = -1 (left bracket endpoint)
        let solver = BrentSolver::default();
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
        let solver = BrentSolver::new(1e-12, 100);
        let result = solver.find_root(|x| (x - 1.0).powi(3), 0.0, 2.0, None, None);
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
        let solver = BrentSolver::default();
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
        let solver = BrentSolver::default();
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
        let solver = BrentSolver::default();

        // f(x) = x^3 - x - 2, has a root near 1.52
        let result = solver.find_root(|x| x.powi(3) - x - 2.0, 1.0, 2.0, None, None);

        let (root, f_root, iters) = result.unwrap();

        // Verify it's actually a root
        assert!(f_root.abs() < 1e-12);
        // Verify the root value
        assert!((root - 1.5213797).abs() < 1e-6);
        println!("Cubic root found: {} in {} iterations", root, iters);
    }
}
