//! Step observer — inspecting integration progress.
//!
//! Integrates a Lotka-Volterra predator-prey model and uses a `StepObserver`
//! to record the trajectory at each accepted step without dense output overhead.
//!
//! Run with:
//!   cargo run --example step_observer

use rkf78::{IntegrationConfig, OdeSystem, Rkf78, Scalar, StepObserver, Tolerances};

/// Lotka-Volterra predator-prey model:
///   dx/dt =  αx - βxy     (prey growth - predation)
///   dy/dt = δxy - γy       (predator growth - natural death)
struct LotkaVolterra {
    alpha: f64, // prey birth rate
    beta: f64,  // predation rate
    delta: f64, // predator growth from predation
    gamma: f64, // predator death rate
}

/// State vector: [prey, predator]
impl OdeSystem<f64, 2> for LotkaVolterra {
    fn rhs(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
        let prey = y[0];
        let predator = y[1];
        dydt[0] = self.alpha * prey - self.beta * prey * predator;
        dydt[1] = self.delta * prey * predator - self.gamma * predator;
    }
}

/// Records the state at each accepted step.
struct TrajectoryRecorder {
    times: Vec<f64>,
    states: Vec<[f64; 2]>,
    max_error: f64,
}

impl TrajectoryRecorder {
    fn new() -> Self {
        Self {
            times: Vec::new(),
            states: Vec::new(),
            max_error: 0.0,
        }
    }
}

impl<T: Scalar<Real = f64>> StepObserver<T, 2> for TrajectoryRecorder {
    fn on_step(&mut self, t: f64, y: &[T; 2], _h: f64, error: f64) {
        // Record time and state (converting from generic T to f64)
        self.times.push(t);
        self.states.push([y[0].norm(), y[1].norm()]);
        self.max_error = self.max_error.max(error);
    }
}

fn main() {
    // Classic Lotka-Volterra parameters
    let sys = LotkaVolterra {
        alpha: 1.1,
        beta: 0.4,
        delta: 0.1,
        gamma: 0.4,
    };

    let y0 = [10.0, 10.0]; // 10 prey, 10 predators
    let tf = 50.0;

    let tol = Tolerances::new(1e-10, 1e-10);
    let mut solver = Rkf78::new(tol);
    let config = IntegrationConfig::new(0.0, tf, 0.1);

    let mut recorder = TrajectoryRecorder::new();
    let (t_final, y_final) = solver
        .integrate_with_observer(&sys, &config, &y0, &mut recorder)
        .unwrap();

    println!("Lotka-Volterra Predator-Prey Model");
    println!(
        "  α={}, β={}, δ={}, γ={}",
        sys.alpha, sys.beta, sys.delta, sys.gamma
    );
    println!("  Initial: prey={:.1}, predators={:.1}", y0[0], y0[1]);
    println!(
        "  Final:   prey={:.4}, predators={:.4} at t={t_final:.2}",
        y_final[0], y_final[1]
    );
    println!();
    println!("  Steps recorded: {}", recorder.times.len());
    println!("  Max error norm: {:.3e}", recorder.max_error);
    println!("  Accepted steps: {}", solver.stats.accepted_steps);
    println!("  Rejected steps: {}", solver.stats.rejected_steps);
    println!();

    // Print a sampled trajectory (every ~10th point)
    let skip = (recorder.times.len() / 15).max(1);
    println!("{:<8} {:<12} {:<12}", "t", "Prey", "Predator");
    println!("{}", "-".repeat(32));
    for (i, (t, state)) in recorder
        .times
        .iter()
        .zip(recorder.states.iter())
        .enumerate()
    {
        if i % skip == 0 || i == recorder.times.len() - 1 {
            println!("{:<8.2} {:<12.4} {:<12.4}", t, state[0], state[1]);
        }
    }
}
