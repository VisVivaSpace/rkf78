//! GPU integration tests — compare GPU batch propagator against CPU reference.
//!
//! These tests require a GPU and the `gpu` feature:
//!   cargo test --features gpu

#![cfg(feature = "gpu")]

use rkf78::gpu::{GpuBatchPropagator, GpuIntegrationParams, GpuState};
use rkf78::{OdeSystem, Rkf78, Tolerances};

/// Earth gravitational parameter [km³/s²]
const MU: f64 = 398600.4418;

/// Create a circular orbit initial state at given altitude.
fn circular_orbit_state(r: f32) -> GpuState {
    let mu_f32 = MU as f32;
    let v = (mu_f32 / r).sqrt();
    GpuState {
        position: [r, 0.0, 0.0],
        velocity: [0.0, v, 0.0],
        epoch: 0.0,
        _pad: 0.0,
    }
}

/// CPU two-body system for reference integration.
struct TwoBody {
    mu: f64,
}

impl OdeSystem<6> for TwoBody {
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

/// Default integration params for a circular LEO orbit (one period).
fn leo_params(t_final: f32) -> GpuIntegrationParams {
    GpuIntegrationParams {
        mu: MU as f32,
        t_final,
        h_init: 60.0,
        h_min: 1e-4,
        h_max: 600.0,
        rtol: 1e-6,
        atol_pos: 1e-3,
        atol_vel: 1e-6,
        max_steps_per_dispatch: 10000,
        _pad: [0; 3],
    }
}

/// Compute orbital period for circular orbit at radius r [km].
fn orbital_period(r: f64) -> f64 {
    2.0 * std::f64::consts::PI * (r.powi(3) / MU).sqrt()
}

// ─── Test 1: Circular orbit GPU vs CPU ─────────────────────────────────

#[test]
fn test_circular_orbit_gpu_vs_cpu() {
    let r0: f64 = 6878.0; // 500 km altitude
    let period = orbital_period(r0) as f32;

    // GPU propagation
    let propagator = GpuBatchPropagator::new();
    let state = circular_orbit_state(r0 as f32);
    let params = leo_params(period);
    let (gpu_states, gpu_statuses) = propagator.propagate_batch(&[state], &params);

    assert_eq!(gpu_statuses[0].status, 1, "GPU trajectory should complete");

    // CPU reference
    let sys = TwoBody { mu: MU };
    let v0 = (MU / r0).sqrt();
    let y0 = [r0, 0.0, 0.0, 0.0, v0, 0.0];
    let tol = Tolerances::new(1e-12, 1e-12);
    let mut solver = Rkf78::new(tol);
    let (_, y_cpu) = solver
        .integrate(&sys, 0.0, &y0, period as f64, 60.0)
        .unwrap();

    // Compare positions — f32 limit is ~7 significant digits on ~7000 km values
    // so expect ~0.1 km accuracy at best
    let dx = (gpu_states[0].position[0] as f64 - y_cpu[0]).abs();
    let dy = (gpu_states[0].position[1] as f64 - y_cpu[1]).abs();
    let dz = (gpu_states[0].position[2] as f64 - y_cpu[2]).abs();
    let pos_err = (dx * dx + dy * dy + dz * dz).sqrt();

    println!(
        "GPU vs CPU position error: {:.1} km (GPU steps: {}, rejected: {})",
        pos_err, gpu_statuses[0].steps, gpu_statuses[0].rejected
    );

    // 100 km tolerance (f32 accumulation over one orbit)
    assert!(
        pos_err < 100.0,
        "Position error {:.1} km exceeds 100 km threshold",
        pos_err
    );
}

// ─── Test 2: Batch independence ────────────────────────────────────────

#[test]
fn test_batch_independence() {
    let r0 = 6878.0f32;
    let period = orbital_period(r0 as f64) as f32;

    let propagator = GpuBatchPropagator::new();
    let state = circular_orbit_state(r0);
    let params = leo_params(period);

    // Propagate 100 identical states
    let states: Vec<GpuState> = vec![state; 100];
    let (gpu_states, gpu_statuses) = propagator.propagate_batch(&states, &params);

    // All should complete
    for (i, s) in gpu_statuses.iter().enumerate() {
        assert_eq!(s.status, 1, "Trajectory {} should complete", i);
    }

    // All results should be identical (bitwise)
    let ref_state = &gpu_states[0];
    for (i, s) in gpu_states.iter().enumerate().skip(1) {
        assert_eq!(
            s.position, ref_state.position,
            "Trajectory {} position differs from trajectory 0",
            i
        );
        assert_eq!(
            s.velocity, ref_state.velocity,
            "Trajectory {} velocity differs from trajectory 0",
            i
        );
        assert_eq!(
            s.epoch, ref_state.epoch,
            "Trajectory {} epoch differs from trajectory 0",
            i
        );
    }
}

// ─── Test 3: Energy conservation ───────────────────────────────────────

#[test]
fn test_energy_conservation_gpu() {
    let r0 = 6878.0f32;
    let period = orbital_period(r0 as f64) as f32;

    let propagator = GpuBatchPropagator::new();
    let state = circular_orbit_state(r0);
    let params = leo_params(period);

    let mu = MU as f32;
    let compute_energy = |s: &GpuState| -> f32 {
        let r = (s.position[0].powi(2) + s.position[1].powi(2) + s.position[2].powi(2)).sqrt();
        let v2 = s.velocity[0].powi(2) + s.velocity[1].powi(2) + s.velocity[2].powi(2);
        0.5 * v2 - mu / r
    };

    let e0 = compute_energy(&state);
    let (gpu_states, _) = propagator.propagate_batch(&[state], &params);
    let e_final = compute_energy(&gpu_states[0]);

    let rel_err = ((e_final - e0) / e0).abs();
    println!("GPU energy conservation: relative drift = {:.3e}", rel_err);

    // f32 limit: expect < 1e-4 relative energy drift over one orbit
    assert!(
        rel_err < 1e-4,
        "Energy drift {:.3e} exceeds 1e-4 threshold",
        rel_err
    );
}

// ─── Test 4: Elliptical orbit GPU vs CPU ───────────────────────────────

#[test]
fn test_elliptical_orbit_gpu_vs_cpu() {
    let rp: f64 = 6678.0; // 300 km periapsis
    let e = 0.5; // eccentricity
    let a = rp / (1.0 - e);
    let ra = a * (1.0 + e);
    let period = 2.0 * std::f64::consts::PI * (a.powi(3) / MU).sqrt();

    // Start at periapsis
    let v_peri = (MU * (2.0 / rp - 1.0 / a)).sqrt();

    // GPU
    let propagator = GpuBatchPropagator::new();
    let gpu_state = GpuState {
        position: [rp as f32, 0.0, 0.0],
        velocity: [0.0, v_peri as f32, 0.0],
        epoch: 0.0,
        _pad: 0.0,
    };
    let params = leo_params(period as f32);
    let (gpu_states, gpu_statuses) = propagator.propagate_batch(&[gpu_state], &params);

    assert_eq!(gpu_statuses[0].status, 1, "GPU trajectory should complete");

    // CPU reference
    let sys = TwoBody { mu: MU };
    let y0 = [rp, 0.0, 0.0, 0.0, v_peri, 0.0];
    let tol = Tolerances::new(1e-12, 1e-12);
    let mut solver = Rkf78::new(tol);
    let (_, y_cpu) = solver.integrate(&sys, 0.0, &y0, period, 10.0).unwrap();

    let dx = (gpu_states[0].position[0] as f64 - y_cpu[0]).abs();
    let dy = (gpu_states[0].position[1] as f64 - y_cpu[1]).abs();
    let dz = (gpu_states[0].position[2] as f64 - y_cpu[2]).abs();
    let pos_err = (dx * dx + dy * dy + dz * dz).sqrt();

    println!(
        "Elliptical orbit (e={}) GPU vs CPU: position error = {:.1} km (apoapsis = {:.0} km)",
        e, pos_err, ra
    );

    // Elliptical orbits accumulate more error; allow up to 500 km
    assert!(
        pos_err < 500.0,
        "Position error {:.1} km exceeds 500 km threshold",
        pos_err
    );
}

// ─── Test 5: Step rejection ────────────────────────────────────────────

#[test]
fn test_step_rejection_gpu() {
    let r0 = 6878.0f32;
    let period = orbital_period(r0 as f64) as f32;

    let propagator = GpuBatchPropagator::new();
    let state = circular_orbit_state(r0);

    // Use a very large initial step size to force rejections
    let mut params = leo_params(period);
    params.h_init = 5000.0; // much larger than orbital period / 10

    let (_, gpu_statuses) = propagator.propagate_batch(&[state], &params);

    assert_eq!(gpu_statuses[0].status, 1, "Should still complete");
    assert!(
        gpu_statuses[0].rejected > 0,
        "Expected step rejections with h_init=5000, got rejected={}",
        gpu_statuses[0].rejected
    );
    println!(
        "Step rejection test: {} accepted, {} rejected",
        gpu_statuses[0].steps, gpu_statuses[0].rejected
    );
}

// ─── Test 6: Multi-dispatch completion ─────────────────────────────────

#[test]
fn test_multi_dispatch_completion() {
    let r0 = 6878.0f32;
    let period = orbital_period(r0 as f64) as f32;

    let propagator = GpuBatchPropagator::new();
    let state = circular_orbit_state(r0);

    // Use very small max_steps_per_dispatch to force multiple dispatches
    let mut params = leo_params(period);
    params.max_steps_per_dispatch = 10;

    let (gpu_states, gpu_statuses) = propagator.propagate_batch(&[state], &params);

    assert_eq!(
        gpu_statuses[0].status, 1,
        "Should complete even with small max_steps_per_dispatch"
    );

    // Epoch should be at t_final
    let epoch_err = (gpu_states[0].epoch - period).abs();
    assert!(
        epoch_err < 1.0,
        "Final epoch {} should be near period {}",
        gpu_states[0].epoch,
        period
    );

    println!(
        "Multi-dispatch test: {} total steps across multiple dispatches",
        gpu_statuses[0].steps
    );
}
