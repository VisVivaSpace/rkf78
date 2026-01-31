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
//! fn compute_rhs(pos: vec3<f32>, vel: vec3<f32>, mu: f32) -> Deriv
//! ```
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

/// GPU batch propagator for parallel trajectory integration.
///
/// Wraps a wgpu compute pipeline and provides a synchronous API for
/// propagating batches of trajectories on the GPU.
///
/// The force model is user-supplied as a WGSL string containing a
/// `compute_rhs` function. Example (Keplerian two-body):
///
/// ```ignore
/// let two_body_wgsl = r#"
/// fn compute_rhs(pos: vec3<f32>, vel: vec3<f32>, mu: f32) -> Deriv {
///     let r2 = dot(pos, pos);
///     let r  = sqrt(r2);
///     let r3 = r2 * r;
///     var d: Deriv;
///     d.dp = vel;
///     d.dv = -mu / r3 * pos;
///     return d;
/// }
/// "#;
/// let propagator = GpuBatchPropagator::new(two_body_wgsl);
/// ```
pub struct GpuBatchPropagator {
    pipeline: Rkf78GpuPipeline,
}

impl GpuBatchPropagator {
    /// Create a new GPU batch propagator with a user-supplied force model.
    ///
    /// # Arguments
    /// * `force_model_wgsl` — WGSL source defining `fn compute_rhs(pos: vec3<f32>, vel: vec3<f32>, mu: f32) -> Deriv`
    ///
    /// Initializes the wgpu instance, adapter, device, and compiles the
    /// compute shader. Panics if no suitable GPU is found.
    pub fn new(force_model_wgsl: &str) -> Self {
        Self {
            pipeline: Rkf78GpuPipeline::new(force_model_wgsl),
        }
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
    ///
    /// # Returns
    /// `(final_states, statuses)` — one entry per trajectory
    pub fn propagate_batch(
        &self,
        initial_states: &[GpuState],
        params: &GpuIntegrationParams,
    ) -> (Vec<GpuState>, Vec<TrajectoryStatus>) {
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
                buffers::read_buffer(device, queue, &status_buffer, n);

            let all_done = statuses.iter().all(|s| s.status == 1 || s.status == 2);
            if all_done {
                // Read final states and return
                let final_states: Vec<GpuState> =
                    buffers::read_buffer(device, queue, &current_buffer, n);
                return (final_states, statuses);
            }
        }

        // If we get here, read whatever we have
        let final_states: Vec<GpuState> = buffers::read_buffer(device, queue, &current_buffer, n);
        let final_statuses: Vec<TrajectoryStatus> =
            buffers::read_buffer(device, queue, &status_buffer, n);
        (final_states, final_statuses)
    }
}

use bytemuck::Zeroable;

impl TrajectoryStatus {
    fn zeroed() -> Self {
        Zeroable::zeroed()
    }
}
