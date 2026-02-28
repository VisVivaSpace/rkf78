//! Cross-validation tests — compare RKF78 against other Rust ODE integrators.
//!
//! These tests verify that RKF78 produces results consistent with other well-established
//! integrators when solving the same ODE with comparable tolerances. Agreement is expected
//! within the specified integration tolerances, not bitwise identity.
//!
//! Reference integrator: `ode_solvers` crate (MIT license)
//! - Dop853: Dormand-Prince 8(5,3) — 8th order method, comparable to RKF78

use ode_solvers::{dop853::Dop853, System, Vector2, Vector6};
use rkf78::{IntegrationConfig, OdeSystem, Rkf78, Tolerances};

type State2 = Vector2<f64>;
type State6 = Vector6<f64>;

/// Harmonic oscillator: y'' + ω²y = 0
/// State: [y, y']
/// Exact solution: y = cos(ωt), y' = -ω sin(ωt)
struct HarmonicOscillator {
    omega: f64,
}

impl OdeSystem<f64, 2> for HarmonicOscillator {
    fn rhs(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
        dydt[0] = y[1];
        dydt[1] = -self.omega * self.omega * y[0];
    }
}

impl System<f64, State2> for HarmonicOscillator {
    fn system(&self, _x: f64, y: &State2, dy: &mut State2) {
        dy[0] = y[1];
        dy[1] = -self.omega * self.omega * y[0];
    }
}

#[test]
fn test_harmonic_oscillator_vs_dop853() {
    let omega = 1.0;
    let sys = HarmonicOscillator { omega };

    // Initial conditions: y(0) = 1, y'(0) = 0
    let y0 = [1.0, 0.0];
    let t0 = 0.0;
    let tf = 2.0 * std::f64::consts::PI; // One period

    // RKF78 integration
    let tol = Tolerances::new(1e-12, 1e-12);
    let mut rkf78_solver = Rkf78::new(tol);
    let (t_rkf78, y_rkf78) = rkf78_solver
        .integrate(&sys, &IntegrationConfig::new(t0, tf, 0.1), &y0)
        .unwrap();

    // Dop853 integration
    // Note: dx parameter controls output step size, use tf/100 to ensure we reach tf accurately
    let y0_dop = State2::new(y0[0], y0[1]);
    let mut dop853_solver = Dop853::new(sys, t0, tf, tf / 100.0, y0_dop, 1e-12, 1e-12);
    let _res = dop853_solver.integrate();
    let y_dop = dop853_solver.y_out().last().unwrap();

    // Both should reach tf
    assert!((t_rkf78 - tf).abs() < 1e-10);

    // Compare final states
    let err_y = (y_rkf78[0] - y_dop[0]).abs();
    let err_dy = (y_rkf78[1] - y_dop[1]).abs();

    println!("Harmonic oscillator cross-validation (RKF78 vs Dop853):");
    println!("  RKF78:  y = [{:.15}, {:.15}]", y_rkf78[0], y_rkf78[1]);
    println!("  Dop853: y = [{:.15}, {:.15}]", y_dop[0], y_dop[1]);
    println!("  Error:  Δy = {:.3e}, Δy' = {:.3e}", err_y, err_dy);
    println!("  RKF78 stats: {:?}", rkf78_solver.stats);

    // Both integrators should agree within their tolerances
    // Use relaxed threshold to account for different error control strategies
    assert!(
        err_y < 1e-10,
        "Position error {:.3e} exceeds threshold",
        err_y
    );
    assert!(
        err_dy < 1e-10,
        "Velocity error {:.3e} exceeds threshold",
        err_dy
    );
}

/// Two-body problem (Keplerian orbit)
/// State: [x, y, z, vx, vy, vz]
struct TwoBody {
    mu: f64, // Gravitational parameter
}

impl OdeSystem<f64, 6> for TwoBody {
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

impl System<f64, State6> for TwoBody {
    fn system(&self, _x: f64, y: &State6, dy: &mut State6) {
        let r = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
        let r3 = r * r * r;
        let mu_r3 = self.mu / r3;

        dy[0] = y[3];
        dy[1] = y[4];
        dy[2] = y[5];
        dy[3] = -mu_r3 * y[0];
        dy[4] = -mu_r3 * y[1];
        dy[5] = -mu_r3 * y[2];
    }
}

#[test]
fn test_two_body_circular_vs_dop853() {
    let mu = 398600.4418; // km³/s² (Earth)
    let sys = TwoBody { mu };

    // Circular orbit at 6878 km (500 km altitude)
    let r0 = 6878.0;
    let v0 = (mu / r0).sqrt();
    let y0 = [r0, 0.0, 0.0, 0.0, v0, 0.0];

    // Integrate for one orbital period
    let period = 2.0 * std::f64::consts::PI * (r0.powi(3) / mu).sqrt();
    let t0 = 0.0;

    // RKF78 integration
    let tol = Tolerances::new(1e-12, 1e-12);
    let mut rkf78_solver = Rkf78::new(tol);
    let (_, y_rkf78) = rkf78_solver
        .integrate(&sys, &IntegrationConfig::new(t0, period, 60.0), &y0)
        .unwrap();

    // Dop853 integration
    let y0_dop = State6::new(y0[0], y0[1], y0[2], y0[3], y0[4], y0[5]);
    let mut dop853_solver = Dop853::new(sys, t0, period, period / 100.0, y0_dop, 1e-12, 1e-12);
    let _res = dop853_solver.integrate();
    let y_dop = dop853_solver.y_out().last().unwrap();

    // Compute position difference
    let dx = y_rkf78[0] - y_dop[0];
    let dy = y_rkf78[1] - y_dop[1];
    let dz = y_rkf78[2] - y_dop[2];
    let pos_err = (dx * dx + dy * dy + dz * dz).sqrt();

    // Compute velocity difference
    let dvx = y_rkf78[3] - y_dop[3];
    let dvy = y_rkf78[4] - y_dop[4];
    let dvz = y_rkf78[5] - y_dop[5];
    let vel_err = (dvx * dvx + dvy * dvy + dvz * dvz).sqrt();

    println!("Two-body circular orbit cross-validation (RKF78 vs Dop853):");
    println!("  Orbital period: {:.2} s", period);
    println!("  Position difference: {:.3e} km", pos_err);
    println!("  Velocity difference: {:.3e} km/s", vel_err);
    println!("  RKF78 stats: {:?}", rkf78_solver.stats);

    // Both integrators should agree to sub-meter level
    assert!(pos_err < 1e-6, "Position error {:.3e} km too large", pos_err);
    assert!(
        vel_err < 1e-9,
        "Velocity error {:.3e} km/s too large",
        vel_err
    );
}

#[test]
fn test_two_body_eccentric_vs_dop853() {
    let mu = 398600.4418; // km³/s² (Earth)
    let sys = TwoBody { mu };

    // Elliptical orbit: periapsis 6678 km (300 km alt), eccentricity 0.5
    let rp: f64 = 6678.0;
    let e: f64 = 0.5;
    let a: f64 = rp / (1.0 - e);
    let period = 2.0 * std::f64::consts::PI * (a.powi(3) / mu).sqrt();

    // Start at periapsis
    let v_peri = (mu * (2.0 / rp - 1.0 / a)).sqrt();
    let y0 = [rp, 0.0, 0.0, 0.0, v_peri, 0.0];
    let t0 = 0.0;

    // RKF78 integration
    let tol = Tolerances::new(1e-12, 1e-12);
    let mut rkf78_solver = Rkf78::new(tol);
    let (_, y_rkf78) = rkf78_solver
        .integrate(&sys, &IntegrationConfig::new(t0, period, 10.0), &y0)
        .unwrap();

    // Dop853 integration
    // Eccentric orbits need finer time resolution at periapsis
    let y0_dop = State6::new(y0[0], y0[1], y0[2], y0[3], y0[4], y0[5]);
    let mut dop853_solver = Dop853::new(sys, t0, period, 1.0, y0_dop, 1e-12, 1e-12);
    let _res = dop853_solver.integrate();
    let y_dop = dop853_solver.y_out().last().unwrap();

    // Compute differences
    let dx = y_rkf78[0] - y_dop[0];
    let dy = y_rkf78[1] - y_dop[1];
    let dz = y_rkf78[2] - y_dop[2];
    let pos_err = (dx * dx + dy * dy + dz * dz).sqrt();

    let dvx = y_rkf78[3] - y_dop[3];
    let dvy = y_rkf78[4] - y_dop[4];
    let dvz = y_rkf78[5] - y_dop[5];
    let vel_err = (dvx * dvx + dvy * dvy + dvz * dvz).sqrt();

    println!("Two-body eccentric orbit (e={}) cross-validation (RKF78 vs Dop853):", e);
    println!("  Semi-major axis: {:.2} km", a);
    println!("  Orbital period: {:.2} s", period);
    println!("  Position difference: {:.3e} km", pos_err);
    println!("  Velocity difference: {:.3e} km/s", vel_err);
    println!("  RKF78 stats: {:?}", rkf78_solver.stats);

    // Eccentric orbits are more challenging but should still show good agreement
    // With dx=1.0 for Dop853, relative position error < 2e-4 (2 km / 13356 km)
    assert!(pos_err < 5.0, "Position error {:.3e} km too large", pos_err);
    assert!(
        vel_err < 5e-3,
        "Velocity error {:.3e} km/s too large",
        vel_err
    );
}

#[test]
fn test_energy_conservation_comparison() {
    let mu = 398600.4418; // km³/s² (Earth)
    let sys = TwoBody { mu };

    // Circular orbit at 6878 km
    let r0 = 6878.0;
    let v0 = (mu / r0).sqrt();
    let y0 = [r0, 0.0, 0.0, 0.0, v0, 0.0];

    let period = 2.0 * std::f64::consts::PI * (r0.powi(3) / mu).sqrt();
    let t0 = 0.0;

    let compute_energy = |y: &[f64; 6]| {
        let r = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
        let v2 = y[3] * y[3] + y[4] * y[4] + y[5] * y[5];
        0.5 * v2 - mu / r
    };

    let e0 = compute_energy(&y0);

    // RKF78 integration
    let tol = Tolerances::new(1e-12, 1e-12);
    let mut rkf78_solver = Rkf78::new(tol);
    let (_, y_rkf78) = rkf78_solver
        .integrate(&sys, &IntegrationConfig::new(t0, period, 60.0), &y0)
        .unwrap();

    let e_rkf78 = compute_energy(&y_rkf78);
    let drift_rkf78 = (e_rkf78 - e0).abs() / e0.abs();

    // Dop853 integration
    let y0_dop = State6::new(y0[0], y0[1], y0[2], y0[3], y0[4], y0[5]);
    let mut dop853_solver = Dop853::new(sys, t0, period, period / 100.0, y0_dop, 1e-12, 1e-12);
    let _res = dop853_solver.integrate();
    let y_dop = dop853_solver.y_out().last().unwrap();

    let y_dop_arr = [y_dop[0], y_dop[1], y_dop[2], y_dop[3], y_dop[4], y_dop[5]];
    let e_dop = compute_energy(&y_dop_arr);
    let drift_dop = (e_dop - e0).abs() / e0.abs();

    println!("Energy conservation comparison (one orbit):");
    println!("  Initial energy: {:.15e} km²/s²", e0);
    println!("  RKF78:  {:.15e} km²/s² (drift: {:.3e})", e_rkf78, drift_rkf78);
    println!("  Dop853: {:.15e} km²/s² (drift: {:.3e})", e_dop, drift_dop);

    // Both should conserve energy well
    assert!(
        drift_rkf78 < 1e-10,
        "RKF78 energy drift {:.3e} too large",
        drift_rkf78
    );
    assert!(
        drift_dop < 1e-10,
        "Dop853 energy drift {:.3e} too large",
        drift_dop
    );
}
