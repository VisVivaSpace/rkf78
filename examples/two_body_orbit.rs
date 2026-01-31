//! Two-body Keplerian orbit — energy conservation check.
//!
//! Propagates a circular LEO orbit for one period and checks that
//! the spacecraft returns to the starting position with conserved energy.
//!
//! Demonstrates per-component tolerances via `Tolerances::with_components()`.
//!
//! Run with:
//!   cargo run --example two_body_orbit

use rkf78::{OdeSystem, Rkf78, Tolerances};

/// Keplerian two-body problem: d²r/dt² = -μ r / |r|³
///
/// State vector: [x, y, z, vx, vy, vz]  (km, km/s)
struct TwoBody {
    mu: f64,
}

impl OdeSystem<6> for TwoBody {
    fn rhs(&self, _t: f64, y: &[f64; 6], dydt: &mut [f64; 6]) {
        let r2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
        let r = r2.sqrt();
        let r3 = r2 * r;
        let mu_r3 = self.mu / r3;

        // dr/dt = v
        dydt[0] = y[3];
        dydt[1] = y[4];
        dydt[2] = y[5];

        // dv/dt = -μ r / |r|³
        dydt[3] = -mu_r3 * y[0];
        dydt[4] = -mu_r3 * y[1];
        dydt[5] = -mu_r3 * y[2];
    }
}

fn energy(mu: f64, y: &[f64; 6]) -> f64 {
    let r = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
    let v2 = y[3] * y[3] + y[4] * y[4] + y[5] * y[5];
    0.5 * v2 - mu / r
}

fn main() {
    let mu = 398600.4418; // Earth μ (km³/s²)
    let sys = TwoBody { mu };

    // Circular orbit at 400 km altitude
    let earth_radius = 6378.137; // km
    let r0 = earth_radius + 400.0;
    let v0 = (mu / r0).sqrt(); // circular velocity

    let y0 = [r0, 0.0, 0.0, 0.0, v0, 0.0];

    // Orbital period: T = 2π √(a³/μ)
    let period = 2.0 * std::f64::consts::PI * (r0.powi(3) / mu).sqrt();

    // Per-component tolerances: tighter on position (km), looser on velocity (km/s)
    let atol = [1e-12, 1e-12, 1e-12, 1e-15, 1e-15, 1e-15];
    let rtol = [1e-13; 6];
    let tol = Tolerances::with_components(atol, rtol);
    let mut solver = Rkf78::new(tol);

    let e0 = energy(mu, &y0);
    let (tf, yf) = solver.integrate(&sys, 0.0, &y0, period, 10.0).unwrap();
    let ef = energy(mu, &yf);

    let pos_err =
        ((yf[0] - y0[0]).powi(2) + (yf[1] - y0[1]).powi(2) + (yf[2] - y0[2]).powi(2)).sqrt();

    println!("Two-Body Circular Orbit");
    println!("  Altitude:  400 km");
    println!("  Radius:    {r0:.3} km");
    println!("  Velocity:  {v0:.6} km/s");
    println!("  Period:    {:.1} s ({:.1} min)", period, period / 60.0);
    println!();
    println!("  Final time: {tf:.6} s");
    println!("  Position error (return to start): {pos_err:.2e} km");
    println!(
        "  Energy drift: {:.2e}  (relative: {:.2e})",
        (ef - e0).abs(),
        ((ef - e0) / e0).abs()
    );
    println!();
    println!("  Accepted steps: {}", solver.stats.accepted_steps);
    println!("  Rejected steps: {}", solver.stats.rejected_steps);
    println!("  Function evals: {}", solver.stats.fn_evals);
}
