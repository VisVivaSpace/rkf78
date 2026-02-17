//! GPU-compatible data structures for batch trajectory propagation.
//!
//! All types use `#[repr(C)]` and `bytemuck` derives for safe GPU buffer casting.
//! Types use f32 for GPU compute performance (~7 significant digits).

use bytemuck::{Pod, Zeroable};

/// GPU-compatible trajectory state (f32 for performance).
///
/// Layout: 32 bytes total (8 × f32).
///
/// Note: f32 provides ~7 significant digits, sufficient for:
/// - Monte Carlo sampling where statistical error dominates
/// - Conjunction screening at km-level accuracy
/// - Real-time visualization
///
/// For high-precision requirements, use the CPU solver.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuState {
    /// Position [km]: x, y, z
    pub position: [f32; 3],
    /// Velocity [km/s]: vx, vy, vz
    pub velocity: [f32; 3],
    /// Current epoch [seconds from reference]
    pub epoch: f32,
    /// Padding for 16-byte alignment
    _pad: f32,
}

impl GpuState {
    /// Create a new GPU state.
    pub fn new(position: [f32; 3], velocity: [f32; 3], epoch: f32) -> Self {
        Self {
            position,
            velocity,
            epoch,
            _pad: 0.0,
        }
    }
}

/// Per-trajectory status and metadata.
///
/// Layout: 16 bytes total (3 × u32 + 1 × f32).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TrajectoryStatus {
    /// 0 = active, 1 = completed, 2 = failed
    pub status: u32,
    /// Number of accepted steps taken
    pub steps: u32,
    /// Number of rejected steps
    pub rejected: u32,
    /// Final step size [seconds]
    pub h_final: f32,
}

/// Integration parameters uniform across the batch.
///
/// Contains only integrator control parameters. Force model parameters
/// (e.g., gravitational parameter, J2 coefficients) are passed separately
/// via a user-defined struct bound at binding 4.
///
/// Layout: 32 bytes total (7 × f32 + 1 × u32).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuIntegrationParams {
    /// Target epoch [seconds from reference]
    pub t_final: f32,
    /// Initial step size [seconds]
    pub h_init: f32,
    /// Minimum step size [seconds]
    pub h_min: f32,
    /// Maximum step size [seconds]
    pub h_max: f32,
    /// Relative tolerance
    pub rtol: f32,
    /// Absolute tolerance for position [km]
    pub atol_pos: f32,
    /// Absolute tolerance for velocity [km/s]
    pub atol_vel: f32,
    /// Maximum integration steps per GPU dispatch
    pub max_steps_per_dispatch: u32,
}

impl GpuIntegrationParams {
    /// Create integration parameters with required fields and sensible defaults.
    ///
    /// Defaults: h_min=1e-6, h_max=3600.0, rtol=1e-6, atol_pos=1e-3, atol_vel=1e-6,
    /// max_steps_per_dispatch=10000.
    pub fn new(t_final: f32, h_init: f32) -> Self {
        Self {
            t_final,
            h_init,
            h_min: 1e-6,
            h_max: 3600.0,
            rtol: 1e-6,
            atol_pos: 1e-3,
            atol_vel: 1e-6,
            max_steps_per_dispatch: 10000,
        }
    }

    /// Set minimum step size [seconds].
    pub fn with_h_min(mut self, v: f32) -> Self {
        self.h_min = v;
        self
    }

    /// Set maximum step size [seconds].
    pub fn with_h_max(mut self, v: f32) -> Self {
        self.h_max = v;
        self
    }

    /// Set relative tolerance.
    pub fn with_rtol(mut self, v: f32) -> Self {
        self.rtol = v;
        self
    }

    /// Set absolute tolerance for position [km].
    pub fn with_atol_pos(mut self, v: f32) -> Self {
        self.atol_pos = v;
        self
    }

    /// Set absolute tolerance for velocity [km/s].
    pub fn with_atol_vel(mut self, v: f32) -> Self {
        self.atol_vel = v;
        self
    }

    /// Set maximum integration steps per GPU dispatch.
    pub fn with_max_steps_per_dispatch(mut self, v: u32) -> Self {
        self.max_steps_per_dispatch = v;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_state_size() {
        assert_eq!(
            std::mem::size_of::<GpuState>(),
            32,
            "GpuState must be 32 bytes for WGSL alignment"
        );
    }

    #[test]
    fn test_trajectory_status_size() {
        assert_eq!(
            std::mem::size_of::<TrajectoryStatus>(),
            16,
            "TrajectoryStatus must be 16 bytes for WGSL alignment"
        );
    }

    #[test]
    fn test_gpu_integration_params_size() {
        assert_eq!(
            std::mem::size_of::<GpuIntegrationParams>(),
            32,
            "GpuIntegrationParams must be 32 bytes for WGSL alignment"
        );
    }

    #[test]
    fn test_bytemuck_round_trip() {
        let state = GpuState::new([6878.0, 0.0, 0.0], [0.0, 7.613, 0.0], 0.0);

        // Cast to bytes and back
        let bytes: &[u8] = bytemuck::bytes_of(&state);
        assert_eq!(bytes.len(), 32);

        let recovered: &GpuState = bytemuck::from_bytes(bytes);
        assert_eq!(recovered.position[0], 6878.0);
        assert_eq!(recovered.velocity[1], 7.613);
        assert_eq!(recovered.epoch, 0.0);
    }
}
