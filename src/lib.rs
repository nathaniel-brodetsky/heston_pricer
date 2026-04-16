use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, StandardNormal};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

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

#[pyfunction]
#[pyo3(signature = (
    s0, k, r, t, v0, kappa, theta, xi, rho,
    n_paths   = 1_000_000,
    n_steps   = 252,
    option_type = "call"
))]
fn price_heston(
    s0: f64,
    k: f64,
    r: f64,
    t: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    xi: f64,
    rho: f64,
    n_paths: usize,
    n_steps: usize,
    option_type: &str,
) -> PyResult<(f64, f64)> {
    if s0 <= 0.0 || k <= 0.0 {
        return Err(PyValueError::new_err("s0 and k must be positive"));
    }
    if v0 < 0.0 {
        return Err(PyValueError::new_err("v0 must be non-negative"));
    }
    if t <= 0.0 {
        return Err(PyValueError::new_err("t must be positive"));
    }
    if rho <= -1.0 || rho >= 1.0 {
        return Err(PyValueError::new_err("rho must be in (-1, 1)"));
    }
    if n_paths == 0 || n_steps == 0 {
        return Err(PyValueError::new_err("n_paths and n_steps must be > 0"));
    }

    let is_call = match option_type.to_lowercase().as_str() {
        "call" => true,
        "put" => false,
        other => {
            return Err(PyValueError::new_err(format!(
                "option_type must be 'call' or 'put', got '{other}'"
            )));
        }
    };

    let params = HestonParams {
        s0,
        v0,
        r,
        kappa,
        theta,
        xi,
        rho,
        t,
        n_steps,
    };
    let discount = (-r * t).exp();

    let (sum_payoff, sum_sq_payoff): (f64, f64) = (0..n_paths)
        .into_par_iter()
        .map(|path_idx| {
            let mut rng = SmallRng::seed_from_u64(path_idx as u64);
            let s_t = simulate_path(params, &mut rng);

            let payoff = if is_call {
                (s_t - k).max(0.0)
            } else {
                (k - s_t).max(0.0)
            };
            (payoff, payoff * payoff)
        })
        .reduce(|| (0.0, 0.0), |(a1, b1), (a2, b2)| (a1 + a2, b1 + b2));

    let n = n_paths as f64;
    let mean = sum_payoff / n;
    let variance = (sum_sq_payoff / n) - (mean * mean);
    let std_err = (variance / n).sqrt();
    let price = discount * mean;
    let price_err = discount * std_err;

    Ok((price, price_err))
}
