# Heston Stochastic-Volatility Options Pricer — Research Notebook

```python
# Cell 0 — environment check
import sys, importlib
print(f"Python {sys.version}")

# The compiled Rust extension lives next to this notebook after `maturin develop`
import heston_pricer
print("✅  heston_pricer Rust module loaded")
print(dir(heston_pricer))
```

---

## 1 — Model Mathematics

### 1.1  The Heston SDEs

The Heston (1993) model specifies two **coupled stochastic differential equations**:

$$
\boxed{
\begin{aligned}
dS_t &= \mu\, S_t\, dt + \sqrt{v_t}\, S_t\, dW_t^{(1)} \\[6pt]
dv_t &= \kappa\bigl(\theta - v_t\bigr)\, dt + \xi\, \sqrt{v_t}\, dW_t^{(2)}
\end{aligned}
}
$$

with the instantaneous correlation constraint:

$$
dW_t^{(1)}\, dW_t^{(2)} = \rho\; dt, \quad \rho \in (-1, 1)
$$

| Parameter | Symbol | Economic Meaning |
|-----------|--------|-----------------|
| Spot price | $S_0$ | Current asset price |
| Initial variance | $v_0$ | $\approx \sigma_0^2$ |
| Risk-free rate | $r$ | Continuously compounded |
| Mean-reversion speed | $\kappa$ | Rate at which $v \to \theta$ |
| Long-run variance | $\theta$ | Steady-state variance |
| Vol-of-vol | $\xi$ | Controls smile curvature |
| Correlation | $\rho$ | Leverage effect ($\rho < 0$ for equities) |

### 1.2  Euler-Maruyama Discretisation (Full-Truncation Scheme)

Partition $[0, T]$ into $N$ steps of width $\Delta t = T/N$.

$$
\begin{aligned}
v_{t+\Delta t} &= v_t + \kappa\bigl(\theta - v_t^+\bigr)\Delta t
                  + \xi\sqrt{v_t^+}\,\sqrt{\Delta t}\; Z_2 \\[6pt]
S_{t+\Delta t} &= S_t \exp\!\left[\left(r - \tfrac{1}{2} v_t^+\right)\Delta t
                  + \sqrt{v_t^+\,\Delta t}\; Z_1\right]
\end{aligned}
$$

where $v^+ = \max(v, 0)$ is the **full-truncation** fix (Lord et al., 2010) and:

$$
\begin{pmatrix} Z_1 \\ Z_2 \end{pmatrix} = 
\begin{pmatrix} 1 & 0 \\ \rho & \sqrt{1-\rho^2} \end{pmatrix}
\begin{pmatrix} \varepsilon_1 \\ \varepsilon_2 \end{pmatrix},
\quad \varepsilon_1, \varepsilon_2 \overset{iid}{\sim} \mathcal{N}(0,1)
$$

### 1.3  Monte Carlo Estimator

The discounted expected payoff (call):

$$
C = e^{-rT}\, \mathbb{E}^{\mathbb{Q}}\!\left[\max\!\left(S_T - K,\; 0\right)\right]
\approx \frac{e^{-rT}}{M} \sum_{i=1}^{M} \max\!\left(S_T^{(i)} - K,\; 0\right)
$$

The **Monte Carlo standard error** shrinks as $\mathcal{O}(M^{-1/2})$:

$$
\text{SE} = \frac{\hat{\sigma}_{\text{payoff}}}{\sqrt{M}}
$$

With $M = 10^6$ paths this gives $\text{SE} \approx \sigma/1000$, typically $\lesssim 5$ cents on a €100 option.

---

## 2 — Basic European Option Pricing

```python
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

# ── Matplotlib style ──────────────────────────────────────────────────────────
rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "font.family":      "monospace",
})

# ── Heston parameters (calibrated to SPX-like surface) ───────────────────────
PARAMS = dict(
    s0    = 100.0,    # spot
    r     = 0.05,     # 5% risk-free
    t     = 1.0,      # 1-year maturity
    v0    = 0.04,     # σ₀ = 20%
    kappa = 2.0,      # mean-reversion speed
    theta = 0.04,     # long-run variance (20% vol)
    xi    = 0.3,      # vol-of-vol
    rho   = -0.7,     # negative leverage effect
)

strikes = [80, 90, 95, 100, 105, 110, 120]

print(f"{'Strike':>8} {'Call Price':>12} {'Std Err':>10} {'Put Price':>12} {'Std Err':>10}")
print("─" * 56)

results = {}
for k in strikes:
    t0 = time.perf_counter()
    call_price, call_err = heston_pricer.price_heston(
        **PARAMS, k=k, n_paths=1_000_000, n_steps=252, option_type="call"
    )
    put_price, put_err = heston_pricer.price_heston(
        **PARAMS, k=k, n_paths=1_000_000, n_steps=252, option_type="put"
    )
    elapsed = time.perf_counter() - t0
    results[k] = (call_price, call_err, put_price, put_err)
    print(f"{k:>8} {call_price:>12.4f} {call_err:>10.4f} "
          f"{put_price:>12.4f} {put_err:>10.4f}   [{elapsed:.2f}s]")
```

---

## 3 — Put-Call Parity Verification

```python
# Put-call parity: C - P = S₀ - K·e^{-rT}
print("\nPut-Call Parity Residuals:")
print(f"{'Strike':>8} {'C - P':>10} {'S - K·e⁻ʳᵀ':>12} {'Residual':>10}")
print("─" * 44)

r, t, s0 = PARAMS["r"], PARAMS["t"], PARAMS["s0"]
for k, (cp, ce, pp, pe) in results.items():
    parity_lhs = cp - pp
    parity_rhs = s0 - k * np.exp(-r * t)
    residual   = abs(parity_lhs - parity_rhs)
    print(f"{k:>8} {parity_lhs:>10.4f} {parity_rhs:>12.4f} {residual:>10.4f}")
```

---

## 4 — Implied Volatility Smile

```python
# ── Generate a dense strike grid ──────────────────────────────────────────────
strike_grid = np.linspace(70, 140, 30)

print(f"Computing IV smile across {len(strike_grid)} strikes via Rust…")
t0 = time.perf_counter()

ivs = heston_pricer.implied_vol_smile(
    s0      = PARAMS["s0"],
    strikes = strike_grid.tolist(),
    r       = PARAMS["r"],
    t       = PARAMS["t"],
    v0      = PARAMS["v0"],
    kappa   = PARAMS["kappa"],
    theta   = PARAMS["theta"],
    xi      = PARAMS["xi"],
    rho     = PARAMS["rho"],
    n_paths = 500_000,
    n_steps = 252,
)
elapsed = time.perf_counter() - t0
ivs_pct = np.array(ivs) * 100
moneyness = np.log(strike_grid / PARAMS["s0"])   # ln(K/S)

print(f"Done in {elapsed:.2f}s  ({elapsed/len(strike_grid)*1000:.0f} ms/strike)")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1117")
fig.suptitle("Heston Model — Implied Volatility Smile  (ρ = −0.7, ξ = 0.3)",
             color="#c9d1d9", fontsize=14, y=1.02)

ACCENT = "#58a6ff"
ACCENT2 = "#f78166"

# Left: IV vs Strike
ax1 = axes[0]
ax1.plot(strike_grid, ivs_pct, color=ACCENT, lw=2.5, label="Heston IV")
ax1.axvline(PARAMS["s0"], color="#8b949e", ls=":", lw=1.2, label=f"ATM (S={PARAMS['s0']})")
ax1.fill_between(strike_grid, ivs_pct, ivs_pct.min(),
                 color=ACCENT, alpha=0.12)
ax1.set_xlabel("Strike (K)")
ax1.set_ylabel("Implied Volatility (%)")
ax1.set_title("IV vs Strike")
ax1.legend(facecolor="#21262d", edgecolor="#30363d")
ax1.grid(True)

# Right: IV vs Log-Moneyness
ax2 = axes[1]
ax2.plot(moneyness, ivs_pct, color=ACCENT2, lw=2.5)
ax2.axvline(0, color="#8b949e", ls=":", lw=1.2, label="ATM (ln(K/S)=0)")
ax2.fill_between(moneyness, ivs_pct, ivs_pct.min(),
                 color=ACCENT2, alpha=0.12)
ax2.set_xlabel("Log-Moneyness  ln(K / S₀)")
ax2.set_ylabel("Implied Volatility (%)")
ax2.set_title("IV vs Log-Moneyness (Smile/Skew)")
ax2.legend(facecolor="#21262d", edgecolor="#30363d")
ax2.grid(True)

plt.tight_layout()
plt.savefig("iv_smile.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("Saved → iv_smile.png")
```

---

## 5 — Barrier Option Pricing

```python
# ── Down-and-out call: barrier at 85% of spot ─────────────────────────────────
barrier_levels = np.linspace(50, 99, 20)
barrier_prices = []
knock_probs    = []

print("Pricing D&O calls across barrier levels…")
for b in barrier_levels:
    price, err, kp = heston_pricer.price_barrier_heston(
        **PARAMS, k=100.0, barrier=b,
        n_paths=500_000, n_steps=252
    )
    barrier_prices.append(price)
    knock_probs.append(kp * 100)

# Vanilla call for reference
vanilla_call, _ = heston_pricer.price_heston(
    **PARAMS, k=100.0, n_paths=1_000_000, n_steps=252, option_type="call"
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor="#0d1117",
                                 sharex=True)
fig.suptitle("Down-and-Out Barrier Call  (K=100, Heston Model)",
             color="#c9d1d9", fontsize=13)

ax1.plot(barrier_levels, barrier_prices, color="#3fb950", lw=2.5,
         label="D&O Call Price")
ax1.axhline(vanilla_call, color=ACCENT, ls="--", lw=1.5,
            label=f"Vanilla Call = {vanilla_call:.3f}")
ax1.fill_between(barrier_levels, barrier_prices, 0,
                 color="#3fb950", alpha=0.15)
ax1.set_ylabel("Option Price")
ax1.legend(facecolor="#21262d", edgecolor="#30363d")
ax1.grid(True)
ax1.set_title("Price vs Barrier Level")

ax2.plot(barrier_levels, knock_probs, color=ACCENT2, lw=2.5)
ax2.fill_between(barrier_levels, knock_probs, 0,
                 color=ACCENT2, alpha=0.15)
ax2.set_xlabel("Barrier Level B")
ax2.set_ylabel("Knock-Out Probability (%)")
ax2.set_title("Knock-Out Probability vs Barrier Level")
ax2.grid(True)

plt.tight_layout()
plt.savefig("barrier_option.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("Saved → barrier_option.png")
```

---

## 6 — Parameter Sensitivity (Greeks via Finite Difference)

```python
def bump_price(param_name: str, bump: float, **base) -> float:
    """Price a call bumping one Heston parameter by `bump`."""
    p = dict(**base)
    p[param_name] += bump
    price, _ = heston_pricer.price_heston(**p, k=100.0, n_paths=500_000,
                                           n_steps=252, option_type="call")
    return price

base = {**PARAMS}
eps  = 1e-4

base_price, _ = heston_pricer.price_heston(
    **base, k=100.0, n_paths=1_000_000, n_steps=252, option_type="call"
)

params_to_bump = {
    "kappa": 0.1,
    "theta": 0.005,
    "xi":    0.01,
    "rho":   0.01,
    "v0":    0.005,
}

print(f"\nBase call price: {base_price:.4f}")
print(f"{'Parameter':>10} {'Δprice/Δparam':>16} {'Interpretation':>28}")
print("─" * 58)

sensitivities = {}
for pname, bump in params_to_bump.items():
    p_up   = bump_price(pname,  bump, **base)
    p_down = bump_price(pname, -bump, **base)
    dPdX   = (p_up - p_down) / (2 * bump)
    sensitivities[pname] = dPdX
    labels = {"kappa":"speed-of-reversion", "theta":"long-run vol²",
              "xi":"vol-of-vol", "rho":"correlation", "v0":"initial vol²"}
    print(f"{pname:>10} {dPdX:>+16.4f}   ({labels[pname]})")

# Bar chart of sensitivities
fig, ax = plt.subplots(figsize=(9, 4), facecolor="#0d1117")
colors = ["#3fb950" if v > 0 else ACCENT2 for v in sensitivities.values()]
bars = ax.bar(sensitivities.keys(), sensitivities.values(), color=colors, width=0.5)
ax.axhline(0, color="#8b949e", lw=1)
ax.set_title("Heston Parameter Sensitivities  (∂Price/∂Param, FD)",
             color="#c9d1d9")
ax.set_ylabel("∂C / ∂param")
ax.set_xlabel("Heston Parameter")
ax.grid(True, axis="y")
plt.tight_layout()
plt.savefig("sensitivities.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("Saved → sensitivities.png")
```

---

## 7 — Convergence Study: Price vs Path Count

```python
path_counts = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
prices, std_errs = [], []

print("Convergence study…")
for n in path_counts:
    p, se = heston_pricer.price_heston(
        **PARAMS, k=100.0, n_paths=n, n_steps=252, option_type="call"
    )
    prices.append(p)
    std_errs.append(se)
    print(f"  n_paths={n:>8,}: price={p:.4f} ± {se:.4f}")

prices   = np.array(prices)
std_errs = np.array(std_errs)
ref      = prices[-1]              # 1M-path estimate as reference

# Theoretical O(1/√M) convergence line
theory_se = std_errs[0] * np.sqrt(path_counts[0]) / np.sqrt(np.array(path_counts))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1117")
fig.suptitle("Monte Carlo Convergence  (Heston ATM Call)",
             color="#c9d1d9", fontsize=13)

ax1.semilogx(path_counts, prices, "o-", color=ACCENT, lw=2, label="MC Price")
ax1.axhline(ref, color="#3fb950", ls="--", lw=1.5, label=f"Reference = {ref:.3f}")
ax1.set_xlabel("Number of Paths (log scale)")
ax1.set_ylabel("Option Price")
ax1.set_title("Price Convergence")
ax1.legend(facecolor="#21262d", edgecolor="#30363d")
ax1.grid(True)

ax2.loglog(path_counts, std_errs, "o-", color=ACCENT2, lw=2, label="MC Std Err")
ax2.loglog(path_counts, theory_se, "--", color="#8b949e", lw=1.5,
           label=r"Theoretical $O(M^{-1/2})$")
ax2.set_xlabel("Number of Paths (log scale)")
ax2.set_ylabel("Standard Error (log scale)")
ax2.set_title(r"Std Error vs Path Count  $[O(M^{-1/2})]$")
ax2.legend(facecolor="#21262d", edgecolor="#30363d")
ax2.grid(True, which="both")

plt.tight_layout()
plt.savefig("convergence.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
```

---

## 8 — Volatility Surface (2D Grid: Strikes × Maturities)

```python
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D   # noqa

maturities   = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
strike_coarse = np.linspace(75, 130, 15)
surface      = np.zeros((len(maturities), len(strike_coarse)))

print("Building vol surface (this takes ~60s for 6 maturities × 15 strikes)…")
for i, T in enumerate(maturities):
    ivs_row = heston_pricer.implied_vol_smile(
        s0=100, strikes=strike_coarse.tolist(), r=0.05,
        t=T, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        n_paths=300_000, n_steps=max(50, int(252 * T)),
    )
    surface[i, :] = np.array(ivs_row) * 100
    print(f"  T={T}: min={surface[i].min():.1f}%  max={surface[i].max():.1f}%")

KK, TT = np.meshgrid(strike_coarse, maturities)

fig = plt.figure(figsize=(12, 7), facecolor="#0d1117")
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#0d1117")
surf = ax.plot_surface(KK, TT, surface,
                        cmap="plasma", edgecolor="none", alpha=0.9)
ax.set_xlabel("Strike K",         labelpad=10)
ax.set_ylabel("Maturity T (yrs)", labelpad=10)
ax.set_zlabel("Implied Vol (%)",  labelpad=10)
ax.set_title("Heston Implied Volatility Surface", pad=20)
fig.colorbar(surf, ax=ax, shrink=0.4, label="IV (%)")
plt.tight_layout()
plt.savefig("vol_surface.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("Saved → vol_surface.png")
```

---

*Notebook generated by: Lead Quant Dev, Derivatives Market Making Desk*  
*Engine: Rust + rayon (parallel MC) → PyO3 → Python*
