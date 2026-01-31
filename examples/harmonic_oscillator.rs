//! Basic RKF78 usage — harmonic oscillator.
//!
//! Integrates y'' + ω²y = 0 for one period and compares with the exact solution.
//!
//! Run with:
//!   cargo run --example harmonic_oscillator

use rkf78::{OdeSystem, Rkf78, Tolerances};

/// Simple harmonic oscillator: y'' + ω²y = 0
///
/// State vector: [y, y']
struct HarmonicOscillator {
    omega: f64,
}

impl OdeSystem<2> for HarmonicOscillator {
    fn rhs(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
        dydt[0] = y[1];
        dydt[1] = -self.omega * self.omega * y[0];
    }
}

fn main() {
    let omega = 2.0;
    let sys = HarmonicOscillator { omega };

    // Integrate for one full period: T = 2π/ω
    let period = 2.0 * std::f64::consts::PI / omega;
    let y0 = [1.0, 0.0]; // y(0) = 1, y'(0) = 0

    let tol = Tolerances::new(1e-12, 1e-12);
    let mut solver = Rkf78::new(tol);

    let (tf, yf) = solver.integrate(&sys, 0.0, &y0, period, 0.01).unwrap();

    // Exact solution: y(t) = cos(ωt), y'(t) = -ω sin(ωt)
    let y_exact = (omega * tf).cos();
    let v_exact = -omega * (omega * tf).sin();

    println!("Harmonic Oscillator (ω = {omega})");
    println!("  Period:      {period:.6} s");
    println!("  Final time:  {tf:.6} s");
    println!();
    println!("  y(T)  = {:.15}   (exact: {:.15})", yf[0], y_exact);
    println!("  y'(T) = {:.15}   (exact: {:.15})", yf[1], v_exact);
    println!();
    println!("  Position error: {:.2e}", (yf[0] - y_exact).abs());
    println!("  Velocity error: {:.2e}", (yf[1] - v_exact).abs());
    println!();
    println!("  Accepted steps: {}", solver.stats.accepted_steps);
    println!("  Rejected steps: {}", solver.stats.rejected_steps);
    println!("  Function evals: {}", solver.stats.fn_evals);
}
