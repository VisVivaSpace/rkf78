// RKF78 Compute Shader — 13-stage adaptive Runge-Kutta-Fehlberg 7(8)
//
// Coefficients from NASA TR R-287, Table X (Fehlberg, 1968).
// Each GPU thread independently propagates one trajectory.
//
// USER-SUPPLIED FUNCTION REQUIRED:
//
//   fn compute_rhs(pos: vec3<f32>, vel: vec3<f32>, mu: f32) -> Deriv
//
// This function computes dy/dt = [velocity, acceleration].
// It is concatenated with this shader at pipeline creation time.
// See examples/gpu_two_body.rs for a Keplerian two-body implementation.
//
// NOTE: Buffer structs use scalar fields (not vec3) to match Rust repr(C)
// layout. WGSL vec3<f32> has 16-byte alignment which would add padding.

// ─── Struct definitions (must match Rust repr(C) types exactly) ─────────

struct State {
    px: f32, py: f32, pz: f32,  // position [km]
    vx: f32, vy: f32, vz: f32,  // velocity [km/s]
    epoch: f32,
    _pad: f32,
}
// 8 × f32 = 32 bytes, alignment 4 — matches Rust GpuState

struct TrajectoryStatus {
    status: u32,     // 0 = active, 1 = completed, 2 = failed
    steps: u32,
    rejected: u32,
    h_final: f32,
}
// 4 × 4 = 16 bytes — matches Rust TrajectoryStatus

struct IntegrationParams {
    mu: f32,
    t_final: f32,
    h_init: f32,
    h_min: f32,
    h_max: f32,
    rtol: f32,
    atol_pos: f32,
    atol_vel: f32,
    max_steps_per_dispatch: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
// 12 × 4 = 48 bytes — matches Rust GpuIntegrationParams

// ─── Buffer bindings ────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read>       initial_states: array<State>;
@group(0) @binding(1) var<storage, read_write> current_states: array<State>;
@group(0) @binding(2) var<storage, read_write> status: array<TrajectoryStatus>;
@group(0) @binding(3) var<uniform>             params: IntegrationParams;

// ─── 8th-order weights (B) ──────────────────────────────────────────────

const B: array<f32, 13> = array<f32, 13>(
    41.0 / 840.0,   // b[0]
    0.0,             // b[1]
    0.0,             // b[2]
    0.0,             // b[3]
    0.0,             // b[4]
    34.0 / 105.0,    // b[5]
    9.0 / 35.0,      // b[6]
    9.0 / 35.0,      // b[7]
    9.0 / 280.0,     // b[8]
    9.0 / 280.0,     // b[9]
    41.0 / 840.0,    // b[10]
    0.0,             // b[11]
    0.0,             // b[12]
);

// ─── Error weights (B - B_hat) ──────────────────────────────────────────

const B_ERR: array<f32, 13> = array<f32, 13>(
     41.0 / 840.0,   // b_err[0]
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,             // b_err[5]  = 0
     0.0,             // b_err[6]  = 0
     0.0,             // b_err[7]  = 0
     0.0,             // b_err[8]  = 0
     0.0,             // b_err[9]  = 0
     41.0 / 840.0,    // b_err[10]
    -41.0 / 840.0,    // b_err[11]
    -41.0 / 840.0,    // b_err[12]
);

// ─── Helpers: convert between scalar struct and vec3 ────────────────────

fn load_pos(s: State) -> vec3<f32> { return vec3<f32>(s.px, s.py, s.pz); }
fn load_vel(s: State) -> vec3<f32> { return vec3<f32>(s.vx, s.vy, s.vz); }

// ─── RHS struct (used by user-supplied compute_rhs) ─────────────────────

struct Deriv {
    dp: vec3<f32>,  // d(pos)/dt = vel
    dv: vec3<f32>,  // d(vel)/dt = accel
}

// ─── 13-stage RKF78 step ────────────────────────────────────────────────
//
// Each stage is unrolled with only non-zero A coefficients.
// k[i] stores the derivative at stage i.

struct StepResult {
    pos: vec3<f32>,
    vel: vec3<f32>,
    error: f32,
}

fn rkf78_step(
    pos: vec3<f32>,
    vel: vec3<f32>,
    h: f32,
    mu: f32,
    atol_pos: f32,
    atol_vel: f32,
    rtol: f32,
) -> StepResult {

    // Stage derivatives
    var kp: array<vec3<f32>, 13>;  // position derivatives (= velocity at stage point)
    var kv: array<vec3<f32>, 13>;  // velocity derivatives (= acceleration at stage point)

    var tp: vec3<f32>;  // temp position
    var tv: vec3<f32>;  // temp velocity

    // ── Stage 0: k0 = f(t, y) ───────────────────────────────────────
    let d0 = compute_rhs(pos, vel, mu);
    kp[0] = d0.dp;
    kv[0] = d0.dv;

    // ── Stage 1: A[1][0] = 2/27 ─────────────────────────────────────
    tp = pos + h * (2.0/27.0) * kp[0];
    tv = vel + h * (2.0/27.0) * kv[0];
    let d1 = compute_rhs(tp, tv, mu);
    kp[1] = d1.dp;
    kv[1] = d1.dv;

    // ── Stage 2: A[2][0] = 1/36, A[2][1] = 1/12 ────────────────────
    tp = pos + h * ((1.0/36.0) * kp[0] + (1.0/12.0) * kp[1]);
    tv = vel + h * ((1.0/36.0) * kv[0] + (1.0/12.0) * kv[1]);
    let d2 = compute_rhs(tp, tv, mu);
    kp[2] = d2.dp;
    kv[2] = d2.dv;

    // ── Stage 3: A[3][0] = 1/24, A[3][2] = 1/8 ─────────────────────
    tp = pos + h * ((1.0/24.0) * kp[0] + (1.0/8.0) * kp[2]);
    tv = vel + h * ((1.0/24.0) * kv[0] + (1.0/8.0) * kv[2]);
    let d3 = compute_rhs(tp, tv, mu);
    kp[3] = d3.dp;
    kv[3] = d3.dv;

    // ── Stage 4: A[4][0] = 5/12, A[4][2] = -25/16, A[4][3] = 25/16
    tp = pos + h * ((5.0/12.0) * kp[0] + (-25.0/16.0) * kp[2] + (25.0/16.0) * kp[3]);
    tv = vel + h * ((5.0/12.0) * kv[0] + (-25.0/16.0) * kv[2] + (25.0/16.0) * kv[3]);
    let d4 = compute_rhs(tp, tv, mu);
    kp[4] = d4.dp;
    kv[4] = d4.dv;

    // ── Stage 5: A[5][0] = 1/20, A[5][3] = 1/4, A[5][4] = 1/5 ─────
    tp = pos + h * ((1.0/20.0) * kp[0] + (1.0/4.0) * kp[3] + (1.0/5.0) * kp[4]);
    tv = vel + h * ((1.0/20.0) * kv[0] + (1.0/4.0) * kv[3] + (1.0/5.0) * kv[4]);
    let d5 = compute_rhs(tp, tv, mu);
    kp[5] = d5.dp;
    kv[5] = d5.dv;

    // ── Stage 6: A[6][0] = -25/108, A[6][3] = 125/108, A[6][4] = -65/27, A[6][5] = 125/54
    tp = pos + h * ((-25.0/108.0) * kp[0] + (125.0/108.0) * kp[3] + (-65.0/27.0) * kp[4] + (125.0/54.0) * kp[5]);
    tv = vel + h * ((-25.0/108.0) * kv[0] + (125.0/108.0) * kv[3] + (-65.0/27.0) * kv[4] + (125.0/54.0) * kv[5]);
    let d6 = compute_rhs(tp, tv, mu);
    kp[6] = d6.dp;
    kv[6] = d6.dv;

    // ── Stage 7: A[7][0] = 31/300, A[7][4] = 61/225, A[7][5] = -2/9, A[7][6] = 13/900
    tp = pos + h * ((31.0/300.0) * kp[0] + (61.0/225.0) * kp[4] + (-2.0/9.0) * kp[5] + (13.0/900.0) * kp[6]);
    tv = vel + h * ((31.0/300.0) * kv[0] + (61.0/225.0) * kv[4] + (-2.0/9.0) * kv[5] + (13.0/900.0) * kv[6]);
    let d7 = compute_rhs(tp, tv, mu);
    kp[7] = d7.dp;
    kv[7] = d7.dv;

    // ── Stage 8: A[8][0]=2, A[8][3]=-53/6, A[8][4]=704/45, A[8][5]=-107/9, A[8][6]=67/90, A[8][7]=3
    tp = pos + h * (2.0 * kp[0] + (-53.0/6.0) * kp[3] + (704.0/45.0) * kp[4] + (-107.0/9.0) * kp[5] + (67.0/90.0) * kp[6] + 3.0 * kp[7]);
    tv = vel + h * (2.0 * kv[0] + (-53.0/6.0) * kv[3] + (704.0/45.0) * kv[4] + (-107.0/9.0) * kv[5] + (67.0/90.0) * kv[6] + 3.0 * kv[7]);
    let d8 = compute_rhs(tp, tv, mu);
    kp[8] = d8.dp;
    kv[8] = d8.dv;

    // ── Stage 9: A[9][0]=-91/108, A[9][3]=23/108, A[9][4]=-976/135, A[9][5]=311/54, A[9][6]=-19/60, A[9][7]=17/6, A[9][8]=-1/12
    tp = pos + h * ((-91.0/108.0) * kp[0] + (23.0/108.0) * kp[3] + (-976.0/135.0) * kp[4] + (311.0/54.0) * kp[5] + (-19.0/60.0) * kp[6] + (17.0/6.0) * kp[7] + (-1.0/12.0) * kp[8]);
    tv = vel + h * ((-91.0/108.0) * kv[0] + (23.0/108.0) * kv[3] + (-976.0/135.0) * kv[4] + (311.0/54.0) * kv[5] + (-19.0/60.0) * kv[6] + (17.0/6.0) * kv[7] + (-1.0/12.0) * kv[8]);
    let d9 = compute_rhs(tp, tv, mu);
    kp[9] = d9.dp;
    kv[9] = d9.dv;

    // ── Stage 10: A[10][0]=2383/4100, A[10][3]=-341/164, A[10][4]=4496/1025, A[10][5]=-301/82, A[10][6]=2133/4100, A[10][7]=45/82, A[10][8]=45/164, A[10][9]=18/41
    tp = pos + h * ((2383.0/4100.0) * kp[0] + (-341.0/164.0) * kp[3] + (4496.0/1025.0) * kp[4] + (-301.0/82.0) * kp[5] + (2133.0/4100.0) * kp[6] + (45.0/82.0) * kp[7] + (45.0/164.0) * kp[8] + (18.0/41.0) * kp[9]);
    tv = vel + h * ((2383.0/4100.0) * kv[0] + (-341.0/164.0) * kv[3] + (4496.0/1025.0) * kv[4] + (-301.0/82.0) * kv[5] + (2133.0/4100.0) * kv[6] + (45.0/82.0) * kv[7] + (45.0/164.0) * kv[8] + (18.0/41.0) * kv[9]);
    let d10 = compute_rhs(tp, tv, mu);
    kp[10] = d10.dp;
    kv[10] = d10.dv;

    // ── Stage 11: A[11][0]=3/205, A[11][5]=-6/41, A[11][6]=-3/205, A[11][7]=-3/41, A[11][8]=3/41, A[11][9]=6/41
    tp = pos + h * ((3.0/205.0) * kp[0] + (-6.0/41.0) * kp[5] + (-3.0/205.0) * kp[6] + (-3.0/41.0) * kp[7] + (3.0/41.0) * kp[8] + (6.0/41.0) * kp[9]);
    tv = vel + h * ((3.0/205.0) * kv[0] + (-6.0/41.0) * kv[5] + (-3.0/205.0) * kv[6] + (-3.0/41.0) * kv[7] + (3.0/41.0) * kv[8] + (6.0/41.0) * kv[9]);
    let d11 = compute_rhs(tp, tv, mu);
    kp[11] = d11.dp;
    kv[11] = d11.dv;

    // ── Stage 12: A[12][0]=-1777/4100, A[12][3]=-341/164, A[12][4]=4496/1025, A[12][5]=-289/82, A[12][6]=2193/4100, A[12][7]=51/82, A[12][8]=33/164, A[12][9]=12/41, A[12][11]=1
    tp = pos + h * ((-1777.0/4100.0) * kp[0] + (-341.0/164.0) * kp[3] + (4496.0/1025.0) * kp[4] + (-289.0/82.0) * kp[5] + (2193.0/4100.0) * kp[6] + (51.0/82.0) * kp[7] + (33.0/164.0) * kp[8] + (12.0/41.0) * kp[9] + 1.0 * kp[11]);
    tv = vel + h * ((-1777.0/4100.0) * kv[0] + (-341.0/164.0) * kv[3] + (4496.0/1025.0) * kv[4] + (-289.0/82.0) * kv[5] + (2193.0/4100.0) * kv[6] + (51.0/82.0) * kv[7] + (33.0/164.0) * kv[8] + (12.0/41.0) * kv[9] + 1.0 * kv[11]);
    let d12 = compute_rhs(tp, tv, mu);
    kp[12] = d12.dp;
    kv[12] = d12.dv;

    // ── 8th-order solution ──────────────────────────────────────────
    var new_pos = pos;
    var new_vel = vel;
    for (var i: u32 = 0u; i < 13u; i++) {
        new_pos = new_pos + h * B[i] * kp[i];
        new_vel = new_vel + h * B[i] * kv[i];
    }

    // ── Error estimate ──────────────────────────────────────────────
    var err_pos = vec3<f32>(0.0, 0.0, 0.0);
    var err_vel = vec3<f32>(0.0, 0.0, 0.0);
    for (var i: u32 = 0u; i < 13u; i++) {
        err_pos = err_pos + B_ERR[i] * kp[i];
        err_vel = err_vel + B_ERR[i] * kv[i];
    }
    err_pos = h * err_pos;
    err_vel = h * err_vel;

    // ── Error norm (infinity norm, scaled) ──────────────────────────
    var max_err: f32 = 0.0;
    for (var i: u32 = 0u; i < 3u; i++) {
        let scale_p = atol_pos + rtol * abs(new_pos[i]);
        max_err = max(max_err, abs(err_pos[i]) / scale_p);

        let scale_v = atol_vel + rtol * abs(new_vel[i]);
        max_err = max(max_err, abs(err_vel[i]) / scale_v);
    }

    var result: StepResult;
    result.pos = new_pos;
    result.vel = new_vel;
    result.error = max_err;
    return result;
}

// ─── Main compute entry point ───────────────────────────────────────────

@compute @workgroup_size(64)
fn propagate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // Bounds check
    if idx >= arrayLength(&current_states) {
        return;
    }

    // Skip completed or failed trajectories
    if status[idx].status != 0u {
        return;
    }

    var pos   = load_pos(current_states[idx]);
    var vel   = load_vel(current_states[idx]);
    var epoch = current_states[idx].epoch;
    var h     = status[idx].h_final;

    // On first dispatch, use h_init
    if h == 0.0 {
        h = params.h_init;
    }

    let mu       = params.mu;
    let t_final  = params.t_final;
    let h_min    = params.h_min;
    let h_max    = params.h_max;
    let rtol     = params.rtol;
    let atol_pos = params.atol_pos;
    let atol_vel = params.atol_vel;

    var steps_this_dispatch: u32 = 0u;

    // Integration loop (bounded by max_steps_per_dispatch to avoid GPU timeout)
    while steps_this_dispatch < params.max_steps_per_dispatch {
        let remaining = t_final - epoch;

        // Check if we're done
        if remaining <= h_min {
            status[idx].status = 1u;  // completed
            break;
        }

        // Don't overshoot
        var h_step = min(h, remaining);
        h_step = clamp(h_step, h_min, h_max);

        // Take a step
        let result = rkf78_step(pos, vel, h_step, mu, atol_pos, atol_vel, rtol);

        if result.error <= 1.0 {
            // Accept step
            pos   = result.pos;
            vel   = result.vel;
            epoch = epoch + h_step;
            status[idx].steps = status[idx].steps + 1u;
        } else {
            // Reject step
            status[idx].rejected = status[idx].rejected + 1u;
        }

        // Step-size control: I-controller
        // h_new = safety * h * error^(-1/8)
        var factor: f32;
        if result.error == 0.0 {
            factor = 5.0;  // max_factor
        } else {
            factor = 0.9 * pow(result.error, -0.125);
            factor = clamp(factor, 0.2, 5.0);
        }
        h = clamp(h * factor, h_min, h_max);

        steps_this_dispatch = steps_this_dispatch + 1u;

        // Check for convergence failure
        if h <= h_min && result.error > 1.0 {
            status[idx].status = 2u;  // failed
            break;
        }
    }

    // Write back state
    current_states[idx].px    = pos.x;
    current_states[idx].py    = pos.y;
    current_states[idx].pz    = pos.z;
    current_states[idx].vx    = vel.x;
    current_states[idx].vy    = vel.y;
    current_states[idx].vz    = vel.z;
    current_states[idx].epoch = epoch;
    status[idx].h_final       = h;
}
