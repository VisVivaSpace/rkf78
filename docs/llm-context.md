# RKF78 — LLM Context Reference

Structured reference for LLM consumption. For full mathematical details, see [algorithm.md](algorithm.md).

---

## Crate Purpose

Zero-dependency Runge-Kutta-Fehlberg 7(8) ODE integrator in Rust. Solves `dy/dt = f(t, y)` with adaptive step-size control and event detection. Designed for spacecraft trajectory propagation.

## Module Layout

| Module | Purpose |
|--------|---------|
| `src/lib.rs` | Public API re-exports, crate-level docs |
| `src/coefficients.rs` | Butcher tableau constants from NASA TR R-287 Table X |
| `src/solver.rs` | Core integrator: `Rkf78<N>`, `OdeSystem<N>`, tolerances, stepping, `integrate()`, `integrate_to_event()` |
| `src/events.rs` | Event detection: `EventFunction<N>`, `BrentSolver`, sign-change monitoring |
| `src/gpu/` | GPU batch propagation via `wgpu` compute shaders (feature-gated: `gpu`) |

## API Surface

### Traits

```rust
// User implements this for their ODE system
pub trait OdeSystem<const N: usize> {
    fn rhs(&self, t: f64, y: &[f64; N], dydt: &mut [f64; N]);
}

// User implements this for event detection
pub trait EventFunction<const N: usize> {
    fn eval(&self, t: f64, y: &[f64; N]) -> f64;
}
```

### Core Struct

```rust
pub struct Rkf78<const N: usize> {
    // Public fields:
    pub h_min: f64,           // Default: 1e-14
    pub h_max: f64,           // Default: f64::INFINITY
    pub max_steps: u64,       // Default: 10_000_000
    pub stats: Stats,         // fn_evals, accepted_steps, rejected_steps
    pub collected_events: Vec<EventResult<N>>,  // Events from EventAction::Continue
}

impl<const N: usize> Rkf78<N> {
    pub fn new(tol: Tolerances<N>) -> Self;
    pub fn set_step_limits(&mut self, h_min: f64, h_max: f64);
    pub fn reset_stats(&mut self);

    pub fn step(&mut self, sys: &S, t: f64, y: &[f64; N], h: f64) -> StepResult<N>;

    pub fn integrate(&mut self, sys: &S, t0: f64, y0: &[f64; N], tf: f64, h0: f64)
        -> Result<(f64, [f64; N]), IntegrationError>;

    pub fn integrate_to_event(&mut self, sys: &S, event: &E, config: &EventConfig,
                              t0: f64, y0: &[f64; N], tf: f64, h0: f64)
        -> Result<IntegrationResult<N>, IntegrationError>;
}
```

### Tolerances

```rust
pub struct Tolerances<const N: usize> {
    pub atol: [f64; N],  // Absolute tolerance per component
    pub rtol: [f64; N],  // Relative tolerance per component
}

impl<const N: usize> Tolerances<N> {
    pub fn new(atol: f64, rtol: f64) -> Self;              // Uniform
    pub fn with_components(atol: [f64; N], rtol: [f64; N]) -> Self;  // Per-component
}
```

### Event Configuration

```rust
pub enum EventDirection { Rising, Falling, Any }
pub enum EventAction { Stop, Continue }

pub struct EventConfig {
    pub direction: EventDirection,  // Default: Any
    pub action: EventAction,        // Default: Stop
    pub root_tol: f64,             // Default: 1e-12
    pub max_iter: usize,           // Default: 50
}
```

### Result & Error Types

```rust
pub enum IntegrationResult<const N: usize> {
    Completed { t: f64, y: [f64; N] },
    Event(EventResult<N>),
}

pub struct EventResult<const N: usize> {
    pub t: f64,
    pub y: [f64; N],
    pub g_value: f64,
    pub iterations: usize,
}

pub enum IntegrationError {
    StepSizeTooSmall { t: f64, h: f64 },
    MaxStepsExceeded,
    EventFindingFailed { message: String },
    InvalidInput { message: String },
    NonFiniteState { t: f64 },
}
```

## Coefficient Structure

- **13 stages**, stored as `[[f64; N]; 13]` workspace
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
let tol = Tolerances::new(1e-12, 1e-12);
let mut solver = Rkf78::new(tol);
let (tf, yf) = solver.integrate(&sys, t0, &y0, tf, h0)?;
```

### Event Detection (Stop)

```rust
let config = EventConfig {
    direction: EventDirection::Rising,
    ..Default::default()
};
match solver.integrate_to_event(&sys, &event, &config, t0, &y0, tf, h0)? {
    IntegrationResult::Event(ev) => { /* ev.t, ev.y */ }
    IntegrationResult::Completed { t, y } => { /* no event */ }
}
```

### Event Detection (Continue)

```rust
let config = EventConfig {
    direction: EventDirection::Any,
    action: EventAction::Continue,
    ..Default::default()
};
let result = solver.integrate_to_event(&sys, &event, &config, t0, &y0, tf, h0)?;
// All detected events in solver.collected_events
```

### Backward Integration

```rust
// Pass negative h0 when tf < t0
let (tf, yf) = solver.integrate(&sys, 10.0, &y0, 0.0, -0.1)?;
```

### GPU Batch Propagation (`gpu` feature)

```rust
use rkf78::gpu::{GpuBatchPropagator, GpuError, GpuState, GpuIntegrationParams, TrajectoryStatus};

// GpuBatchPropagator — propagates many trajectories in parallel on the GPU
pub struct GpuBatchPropagator { /* private */ }
impl GpuBatchPropagator {
    pub fn new(force_model_wgsl: &str) -> Result<Self, GpuError>;
    pub fn propagate_batch(&self, states: &[GpuState], params: &GpuIntegrationParams)
        -> (Vec<GpuState>, Vec<TrajectoryStatus>);
}

// GpuState — 32 bytes, f32 precision (position, velocity, epoch)
// GpuIntegrationParams — 48 bytes (mu, t_final, tolerances, step limits)
// TrajectoryStatus — 16 bytes (status: 0=active/1=completed/2=failed, steps, rejected, h_final)
// GpuError — AdapterNotFound | DeviceCreationFailed(String)
```

## Gotchas

1. **h0 sign must match direction**: `h0` must be positive when `tf > t0`, negative when `tf < t0`
2. **Tolerances must be positive**: `atol > 0` and `rtol >= 0` (both finite)
3. **Event state uses Hermite cubic interpolation**: O(h⁴) accuracy in the event state; event time is found to `root_tol` precision by Brent's method
4. **Not for stiff problems**: explicit method; step size will collapse on stiff systems
5. **collected_events is cleared** at the start of each `integrate_to_event` call
6. **GPU uses f32 precision** (~7 significant digits) vs CPU f64 (~15 digits); GPU energy conservation is ~1e-6 vs CPU ~1e-12

## Tolerance Quick-Reference

| Precision Level | `atol` | `rtol` |
|-----------------|--------|--------|
| High (orbit determination) | `1e-12` | `1e-12` |
| Standard (engineering) | `1e-10` | `1e-10` |
| Fast (screening) | `1e-6` | `1e-6` |
