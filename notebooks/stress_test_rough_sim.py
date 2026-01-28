import numpy as np
import matplotlib.pyplot as plt

from data.fbm import fractional_brownian_motion
from data.rough_vol import rough_volatility
from data.rough_asset import simulate_rough_vol_asset


def run_stress_test(
    label,
    H,
    eta,
    T,
    N,
    n_paths=50,
):
    print(f"\n===== {label} =====")
    print(f"H={H}, eta={eta}, T={T}, N={N}")

    dt = T / N
    S0 = 100.0
    r = 0.05
    v0 = 0.04

    # --- fBM ---
    fbm = fractional_brownian_motion(
        H=H,
        T=T,
        N=N,
        n_paths=n_paths,
        seed=123,
    )

    # --- rough volatility ---
    v = rough_volatility(
        fbm_paths=fbm,
        v0=v0,
        eta=eta,
        T=T,
        H=H,
    )

    # --- Brownian motion for asset ---
    dW = np.random.normal(size=(n_paths, N)) * np.sqrt(dt)

    # --- asset simulation ---
    S = simulate_rough_vol_asset(
        S0=S0,
        r=r,
        v_paths=v,
        dt=dt,
        dW=dW,
    )

    # --- diagnostics ---
    print(f"min(v)   = {v.min():.4e}")
    print(f"max(v)   = {v.max():.4e}")
    print(f"min(S)   = {S.min():.4f}")
    print(f"max(S)   = {S.max():.4f}")
    print(f"mean(S_T)= {S[:, -1].mean():.4f}")

    if not np.isfinite(v).all():
        raise RuntimeError("Volatility contains NaNs or infs")

    if not np.isfinite(S).all():
        raise RuntimeError("Asset price contains NaNs or infs")

    # --- plot one representative path ---
    t = np.linspace(0, T, N + 1)

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(t, S[0], color="tab:blue", label="S(t)")
    ax1.set_ylabel("S(t)")
    ax1.set_xlabel("t")

    ax2 = ax1.twinx()
    ax2.plot(t, v[0], color="tab:red", alpha=0.6, label="v(t)")
    ax2.set_ylabel("v(t)")

    plt.title(label)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":

    # --------------------------------------------------
    # TEST 1: EXTREME ROUGHNESS
    # --------------------------------------------------
    run_stress_test(
        label="Extreme roughness",
        H=0.05,
        eta=1.5,
        T=1.0,
        N=500,
    )

    # --------------------------------------------------
    # TEST 2: LONGER TIME HORIZON
    # --------------------------------------------------
    run_stress_test(
        label="Long horizon",
        H=0.1,
        eta=1.0,
        T=2.0,
        N=1000,
    )

    # --------------------------------------------------
    # TEST 3: VERY HIGH VOL-OF-VOL
    # --------------------------------------------------
    run_stress_test(
        label="High vol-of-vol",
        H=0.1,
        eta=2.0,
        T=1.0,
        N=500,
    )

    print("\nAll stress tests completed successfully.")
