//! Buffer read-back utilities for GPU compute results.

use bytemuck::Pod;

use super::GpuError;

/// Read data from a GPU storage buffer back to the CPU.
///
/// Creates a staging buffer, copies from the source buffer, maps it,
/// and returns the data as a `Vec<T>`.
///
/// # Errors
/// Returns `GpuError::ReadbackFailed` if the buffer mapping or channel communication fails.
pub fn read_buffer<T: Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<T>, GpuError> {
    let byte_size = (count * std::mem::size_of::<T>()) as u64;

    // Create a staging buffer for readback
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Copy from source to staging
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Readback Encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_size);
    queue.submit(Some(encoder.finish()));

    // Map and read
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|e| GpuError::ReadbackFailed(format!("channel closed: {}", e)))?
        .map_err(|e| GpuError::ReadbackFailed(format!("buffer mapping failed: {}", e)))?;

    let data = slice.get_mapped_range();
    let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    Ok(result)
}
