//! GPU batch propagation example — Keplerian two-body orbits.
//!
//! Run with:
//!   cargo run --features gpu --example gpu_two_body

use rkf78::gpu::{GpuBatchPropagator, GpuIntegrationParams, GpuState};

/// User-defined force model parameters (must be 16-byte aligned for WGSL uniform).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TwoBodyParams {
    mu: f32,
    _pad: [f32; 3],
}

/// Keplerian two-body force model in WGSL.
///
/// Declares its own ForceParams struct and reads from @group(0) @binding(4).
const TWO_BODY_WGSL: &str = r#"
struct ForceParams {
    mu: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}
@group(0) @binding(4) var<uniform> force_params: ForceParams;

fn compute_rhs(pos: vec3<f32>, vel: vec3<f32>) -> Deriv {
    let mu = force_params.mu;
    let r2 = dot(pos, pos);
    let r  = sqrt(r2);
    let r3 = r2 * r;
    var d: Deriv;
    d.dp = vel;
    d.dv = -mu / r3 * pos;
    return d;
}
"#;

fn main() {
    let mu: f32 = 398600.4418;
    let force_params = TwoBodyParams { mu, _pad: [0.0; 3] };

    // Create a batch of circular orbits at different altitudes
    let altitudes_km = [200.0_f32, 400.0, 600.0, 800.0, 1000.0];
    let earth_radius: f32 = 6378.0;

    let states: Vec<GpuState> = altitudes_km
        .iter()
        .map(|&alt| {
            let r = earth_radius + alt;
            let v = (mu / r).sqrt();
            GpuState::new([r, 0.0, 0.0], [0.0, v, 0.0], 0.0)
        })
        .collect();

    // Integrate for one LEO orbital period (~90 min)
    let r_ref = earth_radius + 400.0;
    let period = 2.0 * std::f32::consts::PI * (r_ref.powi(3) / mu).sqrt();

    let params = GpuIntegrationParams::new(period, 60.0)
        .with_h_min(1e-4)
        .with_h_max(600.0);

    println!(
        "Propagating {} orbits on GPU for {:.1} s ...",
        states.len(),
        period
    );

    let propagator = GpuBatchPropagator::new(TWO_BODY_WGSL).expect("GPU initialization failed");
    let (final_states, statuses) = propagator
        .propagate_batch(&states, &params, &force_params)
        .expect("GPU propagation failed");

    println!("\nResults:");
    println!(
        "{:<10} {:<12} {:<12} {:<8} {:<8}",
        "Alt (km)", "Final R (km)", "Energy Err", "Steps", "Status"
    );
    println!("{}", "-".repeat(54));

    for (i, alt) in altitudes_km.iter().enumerate() {
        let s = &final_states[i];
        let r = (s.position[0].powi(2) + s.position[1].powi(2) + s.position[2].powi(2)).sqrt();
        let v2 = s.velocity[0].powi(2) + s.velocity[1].powi(2) + s.velocity[2].powi(2);

        let r0 = earth_radius + alt;
        let v0 = (mu / r0).sqrt();
        let e0 = 0.5 * v0 * v0 - mu / r0;
        let ef = 0.5 * v2 - mu / r;
        let rel_err = ((ef - e0) / e0).abs();

        let status_str = match statuses[i].status {
            1 => "done",
            2 => "FAIL",
            _ => "active",
        };

        println!(
            "{:<10.0} {:<12.2} {:<12.3e} {:<8} {:<8}",
            alt, r, rel_err, statuses[i].steps, status_str
        );
    }
}
