//! GPU batch propagation for parallel trajectory integration.
//!
//! This module provides GPU-accelerated batch propagation of Keplerian
//! trajectories using wgpu compute shaders. It supplements (not replaces)
//! the CPU solver — use the CPU solver for single trajectories requiring
//! f64 precision or event detection.
//!
//! Enable with `cargo build --features gpu`.

pub mod types;

pub use types::{GpuIntegrationParams, GpuState, TrajectoryStatus};
