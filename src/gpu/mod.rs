//! GPU batch propagation for parallel trajectory integration.
//!
//! This module provides GPU-accelerated batch propagation of trajectories
//! using wgpu compute shaders. It supplements (not replaces) the CPU
//! solver — use the CPU solver for single trajectories requiring f64
//! precision or event detection.
//!
//! The force model is user-supplied as a WGSL string. You must provide a
//! function with this signature:
//!
//! ```wgsl
//! fn compute_rhs(pos: vec3<f32>, vel: vec3<f32>) -> Deriv
//! ```
//!
//! Force model parameters (e.g., gravitational parameter, J2 coefficients)
//! are passed as a user-defined `#[repr(C)]` struct via the `force_params`
//! argument to [`GpuBatchPropagator::propagate_batch()`]. In your WGSL,
//! declare these at `@group(0) @binding(4)`.
//!
//! See `examples/gpu_two_body.rs` for a Keplerian two-body implementation.
//!
//! Enable with `cargo build --features gpu`.

pub mod buffers;
pub mod pipeline;
pub mod types;

pub use types::{GpuIntegrationParams, GpuState, TrajectoryStatus};

use pipeline::Rkf78GpuPipeline;
use wgpu::util::DeviceExt;

/// Errors from GPU operations.
#[derive(Debug)]
pub enum GpuError {
    /// No suitable GPU adapter was found.
    AdapterNotFound,
    /// Failed to create GPU device.
    DeviceCreationFailed(String),
    /// GPU buffer readback failed.
    ReadbackFailed(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::AdapterNotFound => write!(f, "No suitable GPU adapter found"),
            GpuError::DeviceCreationFailed(msg) => write!(f, "GPU device creation failed: {}", msg),
            GpuError::ReadbackFailed(msg) => write!(f, "GPU buffer readback failed: {}", msg),
        }
    }
}

impl std::error::Error for GpuError {}

/// GPU batch propagator for parallel trajectory integration.
///
/// Wraps a wgpu compute pipeline and provides a synchronous API for
/// propagating batches of trajectories on the GPU.
///
/// The force model is user-supplied as a WGSL string containing a
/// `compute_rhs` function. Force model parameters are declared by the user
/// at `@group(0) @binding(4)`. Example (Keplerian two-body):
///
/// ```ignore
/// let two_body_wgsl = r#"
/// struct ForceParams { mu: f32, _pad0: f32, _pad1: f32, _pad2: f32 }
/// @group(0) @binding(4) var<uniform> force_params: ForceParams;
///
/// fn compute_rhs(pos: vec3<f32>, vel: vec3<f32>) -> Deriv {
///     let mu = force_params.mu;
///     let r2 = dot(pos, pos);
///     let r  = sqrt(r2);
///     let r3 = r2 * r;
///     var d: Deriv;
///     d.dp = vel;
///     d.dv = -mu / r3 * pos;
///     return d;
/// }
/// "#;
/// let propagator = GpuBatchPropagator::new(two_body_wgsl).unwrap();
/// ```
pub struct GpuBatchPropagator {
    pipeline: Rkf78GpuPipeline,
}

impl GpuBatchPropagator {
    /// Create a new GPU batch propagator with a user-supplied force model.
    ///
    /// # Arguments
    /// * `force_model_wgsl` — WGSL source defining `fn compute_rhs(pos: vec3<f32>, vel: vec3<f32>) -> Deriv`
    ///   and a force params struct at `@group(0) @binding(4)`
    ///
    /// # Errors
    /// Returns `GpuError::AdapterNotFound` if no suitable GPU is available,
    /// or `GpuError::DeviceCreationFailed` if the device cannot be created.
    pub fn new(force_model_wgsl: &str) -> Result<Self, GpuError> {
        Ok(Self {
            pipeline: Rkf78GpuPipeline::new(force_model_wgsl)?,
        })
    }

    /// Propagate a batch of trajectories to `params.t_final`.
    ///
    /// Uses multi-dispatch: if not all trajectories complete in one dispatch
    /// (bounded by `max_steps_per_dispatch`), re-dispatches until all are
    /// done or failed.
    ///
    /// # Arguments
    /// * `initial_states` — Starting state for each trajectory
    /// * `params` — Integration parameters (uniform across the batch)
    /// * `force_params` — User-defined force model parameters (`#[repr(C)]`, `Pod`).
    ///   Must be 16-byte aligned (size must be a multiple of 16). This is bound
    ///   at `@group(0) @binding(4)` in the WGSL shader.
    ///
    /// # Returns
    /// `(final_states, statuses)` — one entry per trajectory
    ///
    /// # Errors
    /// Returns `GpuError::ReadbackFailed` if GPU buffer readback fails.
    ///
    /// # Panics
    /// Panics if `size_of::<P>()` is not a multiple of 16 (WGSL uniform alignment).
    pub fn propagate_batch<P: bytemuck::Pod>(
        &self,
        initial_states: &[GpuState],
        params: &GpuIntegrationParams,
        force_params: &P,
    ) -> Result<(Vec<GpuState>, Vec<TrajectoryStatus>), GpuError> {
        assert!(
            std::mem::size_of::<P>().is_multiple_of(16),
            "Force params struct must be 16-byte aligned for WGSL uniform buffer (size {} is not a multiple of 16)",
            std::mem::size_of::<P>()
        );
        let n = initial_states.len();
        let device = &self.pipeline.device;
        let queue = &self.pipeline.queue;

        // Create GPU buffers
        let initial_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Initial States"),
            contents: bytemuck::cast_slice(initial_states),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let current_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Current States"),
            contents: bytemuck::cast_slice(initial_states),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let zero_status = vec![TrajectoryStatus::zeroed(); n];
        let status_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Status"),
            contents: bytemuck::cast_slice(&zero_status),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params"),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let force_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Force Params"),
            contents: bytemuck::bytes_of(force_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RKF78 Bind Group"),
            layout: &self.pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: initial_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: current_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: status_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: force_params_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroup_size = 64usize;
        let num_workgroups = n.div_ceil(workgroup_size) as u32;

        // Multi-dispatch loop: keep dispatching until all trajectories are done
        let max_dispatches = 1000u32;
        for _ in 0..max_dispatches {
            // Dispatch compute
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("RKF78 Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_workgroups, 1, 1);
            }
            queue.submit(Some(encoder.finish()));

            // Read back status to check completion
            let statuses: Vec<TrajectoryStatus> =
                buffers::read_buffer(device, queue, &status_buffer, n)?;

            let all_done = statuses.iter().all(|s| s.status == 1 || s.status == 2);
            if all_done {
                // Read final states and return
                let final_states: Vec<GpuState> =
                    buffers::read_buffer(device, queue, &current_buffer, n)?;
                return Ok((final_states, statuses));
            }
        }

        // If we get here, read whatever we have
        let final_states: Vec<GpuState> = buffers::read_buffer(device, queue, &current_buffer, n)?;
        let final_statuses: Vec<TrajectoryStatus> =
            buffers::read_buffer(device, queue, &status_buffer, n)?;
        Ok((final_states, final_statuses))
    }
}

use bytemuck::Zeroable;

impl TrajectoryStatus {
    fn zeroed() -> Self {
        Zeroable::zeroed()
    }
}
