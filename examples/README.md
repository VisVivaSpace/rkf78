# Examples

Each example is self-contained and demonstrates a specific feature of the crate.

| Example | Feature | Run command |
|---------|---------|-------------|
| [Harmonic Oscillator](harmonic_oscillator.rs) | Basic integration with exact-solution comparison | `cargo run --example harmonic_oscillator` |
| [Dense Output](dense_output.rs) | Continuous trajectory evaluation via `Solution` | `cargo run --example dense_output` |
| [Step Observer](step_observer.rs) | Recording integration progress with `StepObserver` | `cargo run --example step_observer` |
| [Two-Body Orbit](two_body_orbit.rs) | Per-component tolerances and energy conservation | `cargo run --example two_body_orbit` |
| [Event Detection](event_detection.rs) | Zero-crossing detection with Stop and Continue actions | `cargo run --example event_detection` |
| [GPU Two-Body](gpu_two_body.rs) | GPU batch propagation with user-defined force model | `cargo run --features gpu --example gpu_two_body` |

## Suggested reading order

1. **harmonic_oscillator** — Minimal working example: define an `OdeSystem`, set tolerances, call `integrate()`
2. **dense_output** — Record the full trajectory and evaluate at arbitrary times
3. **step_observer** — Inspect each accepted step during integration
4. **event_detection** — Detect zero-crossings of a user-defined function, with both Stop and Continue actions
5. **two_body_orbit** — Per-component tolerances for mixed-unit state vectors
6. **gpu_two_body** — GPU batch propagation with a user-supplied WGSL force model (requires `--features gpu`)
