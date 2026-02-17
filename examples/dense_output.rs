//! Dense output — continuous trajectory evaluation.
//!
//! Integrates a damped oscillator and uses `Solution` to evaluate the
//! trajectory at arbitrary times via Hermite cubic interpolation.
//!
//! Run with:
//!   cargo run --example dense_output

use rkf78::{IntegrationConfig, OdeSystem, Rkf78, Tolerances};

/// Damped harmonic oscillator: y'' + 2ζω y' + ω²y = 0
///
/// State vector: [y, y']
struct DampedOscillator {
    omega: f64,
    zeta: f64,
}

impl OdeSystem<f64, 2> for DampedOscillator {
    fn rhs(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
        dydt[0] = y[1];
        dydt[1] = -2.0 * self.zeta * self.omega * y[1] - self.omega * self.omega * y[0];
    }
}

fn main() {
    let sys = DampedOscillator {
        omega: 2.0,
        zeta: 0.1, // underdamped
    };

    let y0 = [1.0, 0.0]; // y(0) = 1, y'(0) = 0
    let tf = 10.0;

    let tol = Tolerances::new(1e-12, 1e-12);
    let mut solver = Rkf78::new(tol);
    let config = IntegrationConfig::new(0.0, tf, 0.1);

    // integrate_dense records the full trajectory
    let (t_final, y_final, solution) = solver.integrate_dense(&sys, &config, &y0).unwrap();

    println!("Damped Oscillator (ω = {}, ζ = {})", sys.omega, sys.zeta);
    println!("  Integrated to t = {t_final:.6}");
    println!(
        "  Final state: y = {:.6}, y' = {:.6}",
        y_final[0], y_final[1]
    );
    println!("  Solution has {} data points", solution.len());
    println!();

    // Evaluate at evenly-spaced times (these don't need to coincide with integration steps)
    println!("{:<8} {:<14} {:<14}", "t", "y(t)", "y'(t)");
    println!("{}", "-".repeat(36));

    let n_samples = 20;
    for i in 0..=n_samples {
        let t = tf * (i as f64) / (n_samples as f64);
        let y = solution.eval(t).unwrap();
        println!("{:<8.2} {:<14.6} {:<14.6}", t, y[0], y[1]);
    }

    println!();

    // Derivatives are also available via Hermite interpolation
    let t_mid = 5.0;
    let y_mid = solution.eval(t_mid).unwrap();
    let dy_mid = solution.eval_derivative(t_mid).unwrap();
    println!("At t = {t_mid}:");
    println!("  y  = {:.10}", y_mid[0]);
    println!("  y' = {:.10}  (from eval)", y_mid[1]);
    println!("  y' = {:.10}  (from eval_derivative)", dy_mid[0]);
    println!();
    println!("  Accepted steps: {}", solver.stats.accepted_steps);
    println!("  Rejected steps: {}", solver.stats.rejected_steps);
    println!("  Function evals: {}", solver.stats.fn_evals);
}
