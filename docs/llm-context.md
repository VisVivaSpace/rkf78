# RKF78 — LLM Context Reference

Structured reference for LLM consumption. For full mathematical details, see [algorithm.md](algorithm.md).

---

## Crate Purpose

Zero-dependency Runge-Kutta-Fehlberg 7(8) ODE integrator in Rust. Solves `dy/dt = f(t, y)` with adaptive step-size control, event detection, and dense output. Generic over `f32`/`f64`. Optional GPU batch propagation via `wgpu` (feature-gated: `gpu`).

## Module Layout

| Module | Purpose |
|--------|---------|
| `src/lib.rs` | Public API re-exports, crate-level docs |
| `src/scalar.rs` | `Float` and `Scalar` traits for f32/f64 generics |
| `src/coefficients.rs` | Butcher tableau constants from NASA TR R-287 Table X |
| `src/solver.rs` | Core integrator: `Rkf78<T, N>`, `OdeSystem<T, N>`, tolerances, stepping, all `integrate*()` methods |
| `src/events.rs` | Event detection: `EventFunction<T, N>`, `MultiEventFunction<T, N, M>`, Brent's method, Hermite interpolation |
| `src/solution.rs` | Dense output: `Solution<T, N>` with Hermite cubic interpolation |
| `src/gpu/` | GPU batch propagation via `wgpu` compute shaders (feature-gated: `gpu`) |

## Scalar Type System

```rust
/// Real floating-point type (f32 or f64).
/// Used for time, step sizes, tolerances, error estimates.
pub trait Float: Copy + Clone + Debug + Display + PartialOrd + ... + Scalar<Real = Self> {
    const ONE: Self;
    const TWO: Self;
    const HALF: Self;
    const INFINITY: Self;
    fn from_f64(v: f64) -> Self;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn powf(self, exp: Self) -> Self;
    fn is_finite(self) -> bool;
    // ... sin, cos, signum, clamp, max
}

/// Scalar type for state vector components.
/// For real types, identical to Float. Designed for future complex support.
pub trait Scalar: Copy + Clone + Debug + Add + Sub + Mul + AddAssign + MulAssign {
    type Real: Float;
    const ZERO: Self;
    fn from_real(r: Self::Real) -> Self;
    fn norm(self) -> Self::Real;
    fn mul_real(self, r: Self::Real) -> Self;
}

// Implemented for f32 and f64
```

## API Surface

### Traits

```rust
/// User implements this for their ODE system: dy/dt = f(t, y)
pub trait OdeSystem<T: Scalar, const N: usize> {
    fn rhs(&self, t: T::Real, y: &[T; N], dydt: &mut [T; N]);
}

/// User implements this for event detection (single event)
pub trait EventFunction<T: Scalar, const N: usize> {
    fn eval(&self, t: T::Real, y: &[T; N]) -> T::Real;
}

/// User implements this for simultaneous multi-event detection
pub trait MultiEventFunction<T: Scalar, const N: usize, const M: usize> {
    fn eval(&self, t: T::Real, y: &[T; N]) -> [T::Real; M];
}

/// Called after each accepted step during integration
pub trait StepObserver<T: Scalar, const N: usize> {
    fn on_step(&mut self, t: T::Real, y: &[T; N], h: T::Real, error: T::Real);
}
// () implements StepObserver as a no-op
```

### Core Struct

```rust
pub struct Rkf78<T: Scalar, const N: usize> {
    tol: Tolerances<T::Real, N>,          // private
    controller: StepController<T::Real>,   // private
    k: [[T; N]; 13],                       // private workspace
    pub stats: Stats,                      // fn_evals, accepted_steps, rejected_steps
}

impl<T: Scalar, const N: usize> Rkf78<T, N> {
    pub fn new(tol: Tolerances<T::Real, N>) -> Self;
    pub fn with_controller(self, controller: StepController<T::Real>) -> Self;
    pub fn reset_stats(&mut self);

    // Single step (advanced usage)
    pub fn step<S: OdeSystem<T, N>>(&mut self, sys: &S, t: T::Real, y: &[T; N], h: T::Real)
        -> StepResult<T, N>;

    // Main integration methods
    pub fn integrate<S: OdeSystem<T, N>>(
        &mut self, sys: &S, config: &IntegrationConfig<T::Real>, y0: &[T; N],
    ) -> Result<(T::Real, [T; N]), IntegrationError<T::Real>>;

    pub fn integrate_with_observer<S, O>(
        &mut self, sys: &S, config: &IntegrationConfig<T::Real>, y0: &[T; N], observer: &mut O,
    ) -> Result<(T::Real, [T; N]), IntegrationError<T::Real>>
    where S: OdeSystem<T, N>, O: StepObserver<T, N>;

    pub fn integrate_dense<S: OdeSystem<T, N>>(
        &mut self, sys: &S, config: &IntegrationConfig<T::Real>, y0: &[T; N],
    ) -> Result<(T::Real, [T; N], Solution<T, N>), IntegrationError<T::Real>>;

    pub fn integrate_to_event<S, E>(
        &mut self, sys: &S, event: &E, event_config: &EventConfig<T::Real>,
        config: &IntegrationConfig<T::Real>, y0: &[T; N],
    ) -> Result<(IntegrationResult<T, N>, Vec<EventResult<T, N>>), IntegrationError<T::Real>>
    where S: OdeSystem<T, N>, E: EventFunction<T, N>;

    pub fn integrate_with_multi_events<S, E, const M: usize>(
        &mut self, sys: &S, events: &E, event_configs: &[EventConfig<T::Real>; M],
        config: &IntegrationConfig<T::Real>, y0: &[T; N],
    ) -> Result<(IntegrationResult<T, N>, Vec<EventResult<T, N>>), IntegrationError<T::Real>>
    where S: OdeSystem<T, N>, E: MultiEventFunction<T, N, M>;
}
```

### Integration Configuration

```rust
pub struct IntegrationConfig<R: Float> {
    pub t0: R,          // Initial time
    pub tf: R,          // Final time
    pub h0: R,          // Initial step size (positive magnitude; direction inferred)
    pub h_min: R,       // Minimum step size magnitude (default: 1e-14)
    pub h_max: R,       // Maximum step size magnitude (default: infinity)
    pub max_steps: u64,  // Maximum steps (default: 10_000_000)
}

impl<R: Float> IntegrationConfig<R> {
    pub fn new(t0: R, tf: R, h0: R) -> Self;  // h0 stored as abs value
    pub fn with_h_min(self, v: R) -> Self;
    pub fn with_h_max(self, v: R) -> Self;
    pub fn with_max_steps(self, v: u64) -> Self;
}
```

### Tolerances

```rust
pub struct Tolerances<R: Float, const N: usize> {
    pub atol: [R; N],  // Absolute tolerance per component
    pub rtol: [R; N],  // Relative tolerance per component
}

impl<R: Float, const N: usize> Tolerances<R, N> {
    pub fn new(atol: R, rtol: R) -> Self;                        // Uniform
    pub fn with_components(atol: [R; N], rtol: [R; N]) -> Self;  // Per-component
}
```

### Step-Size Controller

```rust
pub struct StepController<R: Float> {
    pub safety: R,       // Safety factor (default: 0.9)
    pub max_factor: R,   // Maximum growth per step (default: 5.0)
    pub min_factor: R,   // Minimum reduction per step (default: 0.2)
    exponent: R,         // private: 1/8 for RKF78
}

impl<R: Float> StepController<R> {
    pub fn new() -> Self;  // defaults
    pub fn with_safety(self, v: R) -> Self;
    pub fn with_max_factor(self, v: R) -> Self;
    pub fn with_min_factor(self, v: R) -> Self;
}
```

### Event Configuration

```rust
pub enum EventDirection { Rising, Falling, Any }  // default: Any
pub enum EventAction { Stop, Continue }            // default: Stop

pub struct EventConfig<R: Float> {
    pub direction: EventDirection,
    pub action: EventAction,
    pub root_tol: R,     // default: 1e-12
    pub max_iter: usize, // default: 50
}
// implements Default
```

### Dense Output

```rust
pub struct Solution<T: Scalar, const N: usize> { /* times, states, derivatives */ }

impl<T: Scalar, const N: usize> Solution<T, N> {
    pub fn eval(&self, t: T::Real) -> Option<[T; N]>;            // Hermite cubic interpolation
    pub fn eval_derivative(&self, t: T::Real) -> Option<[T; N]>; // Derivative interpolation
    pub fn times(&self) -> &[T::Real];
    pub fn states(&self) -> &[[T; N]];
    pub fn derivatives(&self) -> &[[T; N]];
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}
```

### Result & Error Types

```rust
pub enum IntegrationResult<T: Scalar, const N: usize> {
    Completed { t: T::Real, y: [T; N] },
    Event(EventResult<T, N>),
}

pub struct EventResult<T: Scalar, const N: usize> {
    pub t: T::Real,
    pub y: [T; N],
    pub g_value: T::Real,
    pub event_index: usize,  // 0 for single-event methods
    pub iterations: usize,   // Brent iterations used
}

pub enum IntegrationError<R: Float> {
    StepSizeTooSmall { t: R, h: R },
    MaxStepsExceeded,
    EventFindingFailed { message: String },
    InvalidInput { message: String },
    NonFiniteState { t: R },
}

pub struct StepResult<T: Scalar, const N: usize> {
    pub y: [T; N],       // New state (8th order)
    pub t: T::Real,      // New time
    pub error: T::Real,  // Normalized error (≤ 1.0 = accepted)
    pub h_next: T::Real, // Suggested next step size
    pub accepted: bool,
}

pub struct Stats {
    pub fn_evals: u64,
    pub accepted_steps: u64,
    pub rejected_steps: u64,
}
```

## Coefficient Structure

- **13 stages**, stored as `[[T; N]; 13]` workspace
- **8th-order weights** (`B`): nonzero at indices 0, 5, 6, 7, 8, 9, 10
- **7th-order weights** (`B_HAT`): nonzero at indices 5, 6, 7, 8, 9, 11, 12
- **Error weights** (`B_ERR = B - B_HAT`): only 4 nonzero entries at indices 0, 10, 11, 12 — all equal to ±41/840
- Error formula: `TE = (41/840) * h * (k[0] + k[10] - k[11] - k[12])`

## Step-Size Control

- I-controller: `h_new = h * 0.9 * error^(-1/8)`
- Growth bounds: `[0.2×, 5.0×]` per step
- Step accepted when normalized error ≤ 1.0
- Error norm: infinity norm with mixed abs/rel scaling: `max_i(|TE_i| / (atol_i + rtol_i * |y_i|))`

## Common Usage Patterns

### Basic Integration

```rust
use rkf78::{Rkf78, OdeSystem, Tolerances, IntegrationConfig};

struct MySystem;
impl OdeSystem<f64, 2> for MySystem {
    fn rhs(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
        dydt[0] = y[1];
        dydt[1] = -y[0];
    }
}

let tol = Tolerances::new(1e-12, 1e-12);
let mut solver = Rkf78::new(tol);
let config = IntegrationConfig::new(0.0, 10.0, 0.1);
let (tf, yf) = solver.integrate(&MySystem, &config, &[1.0, 0.0]).unwrap();
```

### Dense Output

```rust
let (tf, yf, solution) = solver.integrate_dense(&sys, &config, &y0).unwrap();
let y_at_5 = solution.eval(5.0).unwrap();       // Hermite cubic interpolation
let dy_at_5 = solution.eval_derivative(5.0).unwrap();
```

### Step Observer

```rust
use rkf78::{StepObserver, Scalar};

struct Recorder { times: Vec<f64>, states: Vec<[f64; 2]> }

impl StepObserver<f64, 2> for Recorder {
    fn on_step(&mut self, t: f64, y: &[f64; 2], _h: f64, _error: f64) {
        self.times.push(t);
        self.states.push(*y);
    }
}

let mut rec = Recorder { times: vec![], states: vec![] };
let (tf, yf) = solver.integrate_with_observer(&sys, &config, &y0, &mut rec).unwrap();
// rec.times and rec.states contain the trajectory at accepted steps
```

### Event Detection (Stop)

```rust
use rkf78::{EventFunction, EventConfig, EventDirection, IntegrationResult};

struct ZeroCrossing;
impl EventFunction<f64, 2> for ZeroCrossing {
    fn eval(&self, _t: f64, y: &[f64; 2]) -> f64 { y[0] }
}

let event_config = EventConfig {
    direction: EventDirection::Falling,
    ..Default::default()
};
let int_config = IntegrationConfig::new(0.0, 10.0, 0.1);
let (result, _collected) = solver.integrate_to_event(
    &sys, &ZeroCrossing, &event_config, &int_config, &y0
).unwrap();
match result {
    IntegrationResult::Event(ev) => { /* ev.t, ev.y, ev.g_value, ev.iterations */ }
    IntegrationResult::Completed { t, y } => { /* no event found */ }
}
```

### Event Detection (Continue — collect all crossings)

```rust
use rkf78::EventAction;

let event_config = EventConfig {
    direction: EventDirection::Any,
    action: EventAction::Continue,
    ..Default::default()
};
let (result, collected) = solver.integrate_to_event(
    &sys, &event, &event_config, &int_config, &y0
).unwrap();
// collected: Vec<EventResult<T, N>> — all detected crossings
// result: IntegrationResult::Completed (since action is Continue)
```

### Multi-Event Detection

```rust
use rkf78::MultiEventFunction;

struct TwoEvents;
impl MultiEventFunction<f64, 2, 2> for TwoEvents {
    fn eval(&self, _t: f64, y: &[f64; 2]) -> [f64; 2] {
        [y[0] - 0.5, y[1]]  // event 0: y[0] = 0.5, event 1: y[1] = 0
    }
}

let configs = [
    EventConfig { direction: EventDirection::Falling, ..Default::default() },
    EventConfig { direction: EventDirection::Any, ..Default::default() },
];
let (result, collected) = solver.integrate_with_multi_events(
    &sys, &TwoEvents, &configs, &int_config, &y0
).unwrap();
// EventResult.event_index tells you which event fired
```

### Backward Integration

```rust
// Direction is inferred from tf < t0; h0 is always positive magnitude
let config = IntegrationConfig::new(10.0, 0.0, 0.1);
let (tf, yf) = solver.integrate(&sys, &config, &y0).unwrap();
```

### f32 Integration

```rust
struct MySystemF32;
impl OdeSystem<f32, 2> for MySystemF32 {
    fn rhs(&self, _t: f32, y: &[f32; 2], dydt: &mut [f32; 2]) {
        dydt[0] = y[1];
        dydt[1] = -y[0];
    }
}

let tol = Tolerances::new(1e-6_f32, 1e-6_f32);
let mut solver = Rkf78::<f32, 2>::new(tol);
let config = IntegrationConfig::new(0.0_f32, 10.0_f32, 0.1_f32);
let (tf, yf) = solver.integrate(&MySystemF32, &config, &[1.0_f32, 0.0]).unwrap();
```

### Custom Step Controller

```rust
use rkf78::StepController;

let controller = StepController::new()
    .with_safety(0.85)
    .with_max_factor(3.0)
    .with_min_factor(0.3);

let tol = Tolerances::new(1e-10, 1e-10);
let solver = Rkf78::<f64, 2>::new(tol).with_controller(controller);
```

## GPU Batch Propagation (`gpu` feature)

The GPU module propagates many trajectories in parallel using `wgpu` compute shaders. The GPU solver is `f32`-only and uses the same RKF78 algorithm as the CPU solver.

Users supply their own WGSL force model as a string containing a `compute_rhs` function. Force model parameters are passed as a user-defined `#[repr(C)]` struct bound at `@group(0) @binding(4)`.

```rust
use rkf78::gpu::{GpuBatchPropagator, GpuError, GpuState, GpuIntegrationParams, TrajectoryStatus};

// User-defined force params (must be 16-byte aligned, #[repr(C)], Pod + Zeroable)
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ForceParams {
    mu: f32,
    _pad: [f32; 3],  // pad to 16-byte alignment
}

// WGSL force model — declares its own struct and reads from binding 4
const WGSL: &str = r#"
struct ForceParams { mu: f32, _pad0: f32, _pad1: f32, _pad2: f32 }
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

let propagator = GpuBatchPropagator::new(WGSL)?;

let states = vec![GpuState::new([6778.0, 0.0, 0.0], [0.0, 7.67, 0.0], 0.0)];
let params = GpuIntegrationParams::new(5400.0, 60.0)
    .with_h_min(1e-4)
    .with_h_max(600.0);
let force_params = ForceParams { mu: 398600.4418, _pad: [0.0; 3] };

let (final_states, statuses) = propagator.propagate_batch(&states, &params, &force_params)?;
// statuses[i].status: 0 = active, 1 = completed, 2 = failed
// statuses[i].steps, statuses[i].rejected, statuses[i].h_final
```

### GPU Types

```rust
// 32-byte state (f32 position, velocity, epoch)
pub struct GpuState {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub epoch: f32,
    _pad: f32,  // private, for 16-byte alignment
}
impl GpuState {
    pub fn new(position: [f32; 3], velocity: [f32; 3], epoch: f32) -> Self;
}

// 32-byte integration params (no force model params — those are user-supplied separately)
pub struct GpuIntegrationParams {
    pub t_final: f32,
    pub h_init: f32,
    pub h_min: f32,
    pub h_max: f32,
    pub rtol: f32,
    pub atol_pos: f32,
    pub atol_vel: f32,
    pub max_steps_per_dispatch: u32,
}
impl GpuIntegrationParams {
    pub fn new(t_final: f32, h_init: f32) -> Self;  // sensible defaults
    pub fn with_h_min(self, v: f32) -> Self;
    pub fn with_h_max(self, v: f32) -> Self;
    pub fn with_rtol(self, v: f32) -> Self;
    pub fn with_atol_pos(self, v: f32) -> Self;
    pub fn with_atol_vel(self, v: f32) -> Self;
    pub fn with_max_steps_per_dispatch(self, v: u32) -> Self;
    // validate() is called internally by propagate_batch
}

// 16-byte trajectory status
pub struct TrajectoryStatus {
    pub status: u32,    // 0 = active, 1 = completed, 2 = failed
    pub steps: u32,     // total accepted steps
    pub rejected: u32,  // total rejected steps
    pub h_final: f32,   // final step size
}

pub enum GpuError {
    AdapterNotFound,
    DeviceCreationFailed(String),
    ReadbackFailed(String),
    InvalidParams(String),
    MaxDispatchesExhausted,
}

pub struct GpuBatchPropagator { /* private */ }
impl GpuBatchPropagator {
    pub fn new(force_model_wgsl: &str) -> Result<Self, GpuError>;
    pub fn propagate_batch<P: bytemuck::Pod>(
        &self, states: &[GpuState], params: &GpuIntegrationParams, force_params: &P,
    ) -> Result<(Vec<GpuState>, Vec<TrajectoryStatus>), GpuError>;
}
```

### GPU WGSL Contract

The user-supplied WGSL must define:
```wgsl
fn compute_rhs(pos: vec3<f32>, vel: vec3<f32>) -> Deriv
```

The `Deriv` struct is provided by the engine shader:
```wgsl
struct Deriv { dp: vec3<f32>, dv: vec3<f32> }
```

Force model parameters should be declared at `@group(0) @binding(4)` as a uniform buffer. The Rust-side struct must be `#[repr(C)]`, implement `bytemuck::Pod + bytemuck::Zeroable`, and have a size that is a multiple of 16 bytes.

Bindings 0-3 are reserved by the engine (initial states, current states, status, integration params).

## Gotchas

1. **h0 is always positive**: `IntegrationConfig::new()` stores `h0.abs()`. Direction is inferred from `tf - t0`.
2. **Tolerances must be positive**: `atol > 0` and `rtol >= 0` (both finite), or `InvalidInput` error.
3. **Event state uses Hermite cubic interpolation**: O(h^4) accuracy in the event state; event time is found to `root_tol` precision by Brent's method.
4. **Not for stiff problems**: explicit method; step size will collapse on stiff systems.
5. **GPU uses f32**: ~7 significant digits vs CPU f64 ~15 digits. GPU energy conservation is ~1e-6 vs CPU ~1e-12.
6. **GPU force params must be 16-byte aligned**: `size_of::<P>()` must be a multiple of 16, enforced by `Err(InvalidParams(...))` at runtime.
7. **Events are returned, not stored**: `integrate_to_event` returns `(IntegrationResult, Vec<EventResult>)`. There is no `collected_events` field on the solver.

## Tolerance Quick-Reference

| Precision Level | `atol` | `rtol` |
|-----------------|--------|--------|
| High (reference solutions) | `1e-12` | `1e-12` |
| Standard (engineering) | `1e-10` | `1e-10` |
| Fast (surveys) | `1e-6` | `1e-6` |
