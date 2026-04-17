use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use std::f64::consts::SQRT_2;

#[inline(always)]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[inline(always)]
fn make_seed(k_idx: u64, path_idx: u64) -> u64 {
    splitmix64(k_idx.wrapping_mul(2_654_435_761).wrapping_add(path_idx))
}

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
        }
    }

    (s, knocked)
}

fn bs_call(s: f64, k: f64, r: f64, t: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 || t <= 0.0 {
        return (s - k * (-r * t).exp()).max(0.0);
    }
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;
    s * norm_cdf(d1) - k * (-r * t).exp() * norm_cdf(d2)
}

#[inline]
fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / SQRT_2)
}

fn erfc(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = (0.254_829_592
        + (-0.284_496_736 + (1.421_413_741 + (-1.453_152_027 + 1.061_405_429 * t) * t) * t) * t)
        * t
        * (-(x * x)).exp();
    if x >= 0.0 {
        poly
    } else {
        2.0 - poly
    }
}

fn bs_implied_vol(s: f64, k: f64, r: f64, t: f64, target: f64) -> f64 {
    const SIGMA_LO: f64 = 1e-6;
    const SIGMA_HI_INIT: f64 = 3.0;
    const SIGMA_HI_MAX: f64 = 10.0;
    const TOL: f64 = 1e-7;

    let fwd_intrinsic = (s - k * (-r * t).exp()).max(0.0);
    if target <= fwd_intrinsic + 1e-12 {
        return 0.0;
    }

    let f = |sigma: f64| bs_call(s, k, r, t, sigma) - target;

    let mut a = SIGMA_LO;
    let mut b = SIGMA_HI_INIT;
    let mut fa = f(a);
    let mut fb = f(b);

    while fa * fb > 0.0 && b < SIGMA_HI_MAX {
        b *= 2.0;
        fb = f(b);
    }

    if fa * fb > 0.0 {
        return b;
    }

    let mut c = a;
    let mut fc = fa;
    let mut mflag = true;
    let mut d = 0.0_f64;
    let mut s_pt;

    for _ in 0..100 {
        if (b - a).abs() < TOL {
            break;
        }

        if fa != fc && fb != fc {
            s_pt = a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            s_pt = b - fb * (b - a) / (fb - fa);
        }

        let boundary = (3.0 * a + b) / 4.0;
        let cond1 = if a < b {
            !(boundary <= s_pt && s_pt <= b)
        } else {
            !(b <= s_pt && s_pt <= boundary)
        };
        let cond2 = mflag && (s_pt - b).abs() >= (b - c).abs() / 2.0;
        let cond3 = !mflag && (s_pt - b).abs() >= (c - d).abs() / 2.0;

        if cond1 || cond2 || cond3 {
            s_pt = 0.5 * (a + b);
            mflag = true;
        } else {
            mflag = false;
        }

        let fs = f(s_pt);
        d = c;
        c = b;
        fc = fb;

        if fa * fs < 0.0 {
            b = s_pt;
            fb = fs;
        } else {
            a = s_pt;
            fa = fs;
        }

        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }

    0.5 * (a + b)
}

#[pyfunction]
#[pyo3(signature = (
    s0, k, r, t, v0, kappa, theta, xi, rho,
    n_paths     = 1_000_000,
    n_steps     = 252,
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
            )))
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

    let (sum_payoff, sum_sq): (f64, f64) = (0..n_paths)
        .into_par_iter()
        .map(|path_idx| {
            let mut rng = SmallRng::seed_from_u64(splitmix64(path_idx as u64));
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
    let variance = (sum_sq / n) - (mean * mean);
    let std_err = (variance / n).sqrt();

    Ok((discount * mean, discount * std_err))
}

#[pyfunction]
#[pyo3(signature = (
    s0, k, r, t, v0, kappa, theta, xi, rho, barrier,
    n_paths = 1_000_000,
    n_steps = 252
))]
fn price_barrier_heston(
    s0: f64,
    k: f64,
    r: f64,
    t: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    xi: f64,
    rho: f64,
    barrier: f64,
    n_paths: usize,
    n_steps: usize,
) -> PyResult<(f64, f64, f64)> {
    if barrier >= s0 {
        return Err(PyValueError::new_err(
            "barrier must be strictly below s0 for a down-and-out option",
        ));
    }
    if s0 <= 0.0 || k <= 0.0 || barrier <= 0.0 {
        return Err(PyValueError::new_err("s0, k, barrier must be positive"));
    }

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

    const BARRIER_SEED_OFFSET: u64 = 0x0000_DEAD_BEEF_0000;

    let (sum_payoff, sum_sq, knocked_count): (f64, f64, u64) = (0..n_paths)
        .into_par_iter()
        .map(|path_idx| {
            let seed = splitmix64(path_idx as u64 ^ BARRIER_SEED_OFFSET);
            let mut rng = SmallRng::seed_from_u64(seed);
            let (s_t, knocked) = simulate_path_with_barrier(params, &mut rng, barrier);

            let payoff = if knocked { 0.0 } else { (s_t - k).max(0.0) };
            (payoff, payoff * payoff, knocked as u64)
        })
        .reduce(
            || (0.0, 0.0, 0),
            |(a1, b1, c1), (a2, b2, c2)| (a1 + a2, b1 + b2, c1 + c2),
        );

    let n = n_paths as f64;
    let mean = sum_payoff / n;
    let variance = (sum_sq / n) - (mean * mean);
    let std_err = (variance / n).sqrt();
    let knock_prob = knocked_count as f64 / n;

    Ok((discount * mean, discount * std_err, knock_prob))
}

#[pyfunction]
#[pyo3(signature = (
    s0, strikes, r, t, v0, kappa, theta, xi, rho,
    n_paths = 500_000,
    n_steps = 252
))]
fn implied_vol_smile(
    s0: f64,
    strikes: Vec<f64>,
    r: f64,
    t: f64,
    v0: f64,
    kappa: f64,
    theta: f64,
    xi: f64,
    rho: f64,
    n_paths: usize,
    n_steps: usize,
) -> PyResult<Vec<f64>> {
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

    let ivs: Vec<f64> = strikes
        .iter()
        .enumerate()
        .map(|(k_idx, &k)| {
            let (sum_payoff, sum_sq): (f64, f64) = (0..n_paths)
                .into_par_iter()
                .map(|path_idx| {
                    let seed = make_seed(k_idx as u64, path_idx as u64);
                    let mut rng = SmallRng::seed_from_u64(seed);
                    let s_t = simulate_path(params, &mut rng);
                    let payoff = (s_t - k).max(0.0);
                    (payoff, payoff * payoff)
                })
                .reduce(|| (0.0, 0.0), |(a1, b1), (a2, b2)| (a1 + a2, b1 + b2));

            let n = n_paths as f64;
            let mean_payoff = sum_payoff / n;
            let heston_call = discount * mean_payoff;

            bs_implied_vol(s0, k, r, t, heston_call)
        })
        .collect();

    Ok(ivs)
}

#[pymodule]
fn heston_pricer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(price_heston, m)?)?;
    m.add_function(wrap_pyfunction!(price_barrier_heston, m)?)?;
    m.add_function(wrap_pyfunction!(implied_vol_smile, m)?)?;
    Ok(())
}
