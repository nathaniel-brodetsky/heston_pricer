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
