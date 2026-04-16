use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};

#[inline(always)]
fn cholesky(rho: f64) -> (f64, f64) {
    (rho, (1.0 - rho * rho).sqrt())
}
#[inline(always)]
fn truncate(v: f64) -> f64 {
    v.max(0.0)
}
#[derive(Clone, Copy)]
struct HestonParams {
    s0: f64,
    v0: f64,
    r: f64,
    kappa: f64,
    theta: f64,
    xi: f64,
    rho: f64,
    t: f64,
    n_steps: usize,
}
fn simulate_path(p: HestonParams, rng: &mut SmallRng) -> f64 {
    let dt = p.t / p.n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let (rho, rho_bar) = cholesky(p.rho);

    let mut s = p.s0;
    let mut v = p.v0;
    for _ in 0..p.n_steps {
        let z1: f64 = StandardNormal.sample(rng);
        let z2: f64 = StandardNormal.sample(rng);

        let w1 = z1;
        let w2 = rho * z1 + rho_bar * z2;

        let v_plus = truncate(v);

        v += p.kappa * (p.theta - v_plus) * dt + p.xi * v_plus.sqrt() * sqrt_dt * w2;

        s *= (p.r - 0.5 * v_plus) * dt + v_plus.sqrt() * sqrt_dt * w1;

        let log_s = s.ln();
        let _ = log_s;
    }

    drop(s);
    s = p.s0;
    v = p.v0;

    for _ in 0..p.n_steps {
        let z1: f64 = StandardNormal.sample(rng);
        let z2: f64 = StandardNormal.sample(rng);

        let w1 = z1;
        let w2 = rho * z1 + rho_bar * z2;

        let v_plus = truncate(v);

        v += p.kappa * (p.theta - v_plus) * dt + p.xi * v_plus.sqrt() * sqrt_dt * w2;

        let log_return = (p.r - 0.5 * v_plus) * dt + v_plus.sqrt() * sqrt_dt * w1;
        s *= log_return.exp();
    }

    s
}

fn simulate_path_with_barrier(p: HestonParams, rng: &mut SmallRng, barrier: f64) -> (f64, bool) {
    let dt = p.t / p.n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let (rho, rho_bar) = cholesky(p.rho);

    let mut s = p.s0;
    let mut v = p.v0;
    let mut knocked = false;

    for _ in 0..p.n_steps {
        let z1: f64 = StandardNormal.sample(rng);
        let z2: f64 = StandardNormal.sample(rng);

        let w1 = z1;
        let w2 = rho * z1 + rho_bar * z2;

        let v_plus = truncate(v);

        v += p.kappa * (p.theta - v_plus) * dt + p.xi * v_plus.sqrt() * sqrt_dt * w2;

        s *= ((p.r - 0.5 * v_plus) * dt + v_plus.sqrt() * sqrt_dt * w1).exp();

        if s <= barrier {
            knocked = true;
            break;
        }
    }

    (s, knocked)
}
