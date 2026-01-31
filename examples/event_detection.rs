//! Event detection — periapsis finding in a two-body orbit.
//!
//! Demonstrates both `EventAction::Stop` (halt at first periapsis) and
//! `EventAction::Continue` (collect all periapsis crossings over multiple orbits).
//!
//! Run with:
//!   cargo run --example event_detection

use rkf78::{
    EventAction, EventConfig, EventDirection, EventFunction, IntegrationResult, OdeSystem, Rkf78,
    Tolerances,
};

/// Keplerian two-body problem (same as two_body_orbit example).
struct TwoBody {
    mu: f64,
}

impl OdeSystem<6> for TwoBody {
    fn rhs(&self, _t: f64, y: &[f64; 6], dydt: &mut [f64; 6]) {
        let r2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
        let r = r2.sqrt();
        let r3 = r2 * r;
        let mu_r3 = self.mu / r3;

        dydt[0] = y[3];
        dydt[1] = y[4];
        dydt[2] = y[5];
        dydt[3] = -mu_r3 * y[0];
        dydt[4] = -mu_r3 * y[1];
        dydt[5] = -mu_r3 * y[2];
    }
}

/// Periapsis event: radial velocity r·v = 0, rising means r·v goes negative→positive
/// (i.e., radius is at a minimum).
struct PeriapsisEvent;

impl EventFunction<6> for PeriapsisEvent {
    fn eval(&self, _t: f64, y: &[f64; 6]) -> f64 {
        // r_dot = (r · v) / |r|, but sign is all we need
        y[0] * y[3] + y[1] * y[4] + y[2] * y[5]
    }
}

fn main() {
    let mu = 398600.4418;
    let sys = TwoBody { mu };

    // Elliptical orbit: 400 km × 2000 km altitude
    let earth_radius = 6378.137;
    let r_peri = earth_radius + 400.0;
    let r_apo = earth_radius + 2000.0;
    let a = (r_peri + r_apo) / 2.0;
    let v_peri = (mu * (2.0 / r_peri - 1.0 / a)).sqrt();

    // Start at periapsis, moving prograde
    let y0 = [r_peri, 0.0, 0.0, 0.0, v_peri, 0.0];
    let period = 2.0 * std::f64::consts::PI * (a.powi(3) / mu).sqrt();

    println!("Event Detection — Periapsis Finding");
    println!("  Orbit: {:.0} × {:.0} km altitude", 400.0, 2000.0);
    println!("  Period: {:.1} s ({:.1} min)", period, period / 60.0);
    println!();

    // --- Part 1: Stop at first periapsis after t=0 ---
    // We start at periapsis, so the next one is after one full orbit.
    let event = PeriapsisEvent;
    let config = EventConfig {
        direction: EventDirection::Rising, // r_dot: negative → positive = periapsis
        ..Default::default()
    };

    let tol = Tolerances::new(1e-12, 1e-12);
    let mut solver = Rkf78::new(tol);

    // Integrate for 1.5 periods to guarantee we cross periapsis
    let result = solver
        .integrate_to_event(&sys, &event, &config, 0.0, &y0, 1.5 * period, 10.0)
        .unwrap();

    match result {
        IntegrationResult::Event(ev) => {
            let r = (ev.y[0] * ev.y[0] + ev.y[1] * ev.y[1] + ev.y[2] * ev.y[2]).sqrt();
            println!("Part 1: EventAction::Stop");
            println!("  Periapsis found at t = {:.6} s", ev.t);
            println!(
                "  Radius at periapsis: {:.6} km  (expected: {:.3})",
                r, r_peri
            );
            println!("  Radius error: {:.2e} km", (r - r_peri).abs());
            println!("  Brent iterations: {}", ev.iterations);
        }
        IntegrationResult::Completed { t, .. } => {
            println!("Part 1: No periapsis found (reached t = {t})");
        }
    }

    println!();

    // --- Part 2: Collect all periapsis crossings over 5 orbits ---
    let config_continue = EventConfig {
        direction: EventDirection::Rising,
        action: EventAction::Continue,
        ..Default::default()
    };

    let mut solver2 = Rkf78::new(Tolerances::new(1e-12, 1e-12));
    let _ = solver2
        .integrate_to_event(&sys, &event, &config_continue, 0.0, &y0, 5.0 * period, 10.0)
        .unwrap();

    println!("Part 2: EventAction::Continue (5 orbits)");
    println!(
        "  Found {} periapsis crossings:",
        solver2.collected_events.len()
    );
    for (i, ev) in solver2.collected_events.iter().enumerate() {
        let r = (ev.y[0] * ev.y[0] + ev.y[1] * ev.y[1] + ev.y[2] * ev.y[2]).sqrt();
        println!(
            "    #{}: t = {:10.3} s  r = {:.6} km  err = {:.2e} km",
            i + 1,
            ev.t,
            r,
            (r - r_peri).abs()
        );
    }
}
