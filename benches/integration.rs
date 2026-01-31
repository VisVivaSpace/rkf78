use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rkf78::{OdeSystem, Rkf78, Tolerances};

/// Two-body problem (6-state)
struct TwoBody {
    mu: f64,
}

impl OdeSystem<6> for TwoBody {
    fn rhs(&self, _t: f64, y: &[f64; 6], dydt: &mut [f64; 6]) {
        let r = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
        let r3 = r * r * r;
        let mu_r3 = self.mu / r3;

        dydt[0] = y[3];
        dydt[1] = y[4];
        dydt[2] = y[5];
        dydt[3] = -mu_r3 * y[0];
        dydt[4] = -mu_r3 * y[1];
        dydt[5] = -mu_r3 * y[2];
    }
}

/// Harmonic oscillator (2-state)
struct HarmonicOscillator {
    omega: f64,
}

impl OdeSystem<2> for HarmonicOscillator {
    fn rhs(&self, _t: f64, y: &[f64; 2], dydt: &mut [f64; 2]) {
        dydt[0] = y[1];
        dydt[1] = -self.omega * self.omega * y[0];
    }
}

fn bench_circular_orbit_1period(c: &mut Criterion) {
    let mu: f64 = 398600.4418;
    let r0: f64 = 6878.0;
    let v0 = (mu / r0).sqrt();
    let y0 = [r0, 0.0, 0.0, 0.0, v0, 0.0];
    let period = 2.0 * std::f64::consts::PI * (r0.powi(3) / mu).sqrt();
    let sys = TwoBody { mu };

    c.bench_function("circular_orbit_1period", |b| {
        b.iter(|| {
            let tol = Tolerances::new(1e-12, 1e-12);
            let mut solver = Rkf78::new(tol);
            solver
                .integrate(&sys, 0.0, black_box(&y0), period, 60.0)
                .unwrap()
        })
    });
}

fn bench_harmonic_oscillator_1period(c: &mut Criterion) {
    let omega = 1.0;
    let y0 = [1.0, 0.0];
    let period = 2.0 * std::f64::consts::PI;
    let sys = HarmonicOscillator { omega };

    c.bench_function("harmonic_oscillator_1period", |b| {
        b.iter(|| {
            let tol = Tolerances::new(1e-12, 1e-12);
            let mut solver = Rkf78::new(tol);
            solver
                .integrate(&sys, 0.0, black_box(&y0), period, 0.1)
                .unwrap()
        })
    });
}

criterion_group!(
    benches,
    bench_circular_orbit_1period,
    bench_harmonic_oscillator_1period
);
criterion_main!(benches);
