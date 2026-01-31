//! wgpu compute pipeline setup for RKF78 GPU propagation.

use std::borrow::Cow;

/// Holds the wgpu device, queue, compute pipeline, and bind group layout.
pub struct Rkf78GpuPipeline {
    /// The wgpu device.
    pub device: wgpu::Device,
    /// The wgpu command queue.
    pub queue: wgpu::Queue,
    /// The compiled compute pipeline.
    pub pipeline: wgpu::ComputePipeline,
    /// The bind group layout for buffer bindings.
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl Rkf78GpuPipeline {
    /// Create the pipeline with a user-supplied WGSL force model.
    ///
    /// The `force_model_wgsl` string must define:
    /// ```wgsl
    /// fn compute_rhs(pos: vec3<f32>, vel: vec3<f32>, mu: f32) -> Deriv
    /// ```
    /// This is prepended to the RKF78 engine shader at pipeline creation time.
    ///
    /// Uses `pollster::block_on` for synchronous initialization.
    pub fn new(force_model_wgsl: &str) -> Self {
        pollster::block_on(Self::new_async(force_model_wgsl.to_owned()))
    }

    async fn new_async(force_model_wgsl: String) -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("Failed to find a suitable GPU adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("RKF78 Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("Failed to create GPU device");

        let engine = include_str!("shader.wgsl");
        let full_shader = format!("{}\n{}", force_model_wgsl, engine);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RKF78 Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(full_shader)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RKF78 Bind Group Layout"),
            entries: &[
                // binding 0: initial_states (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: current_states (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: status (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RKF78 Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RKF78 Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("propagate"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
        }
    }
}
