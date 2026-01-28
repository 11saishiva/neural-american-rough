import numpy as np
import matplotlib.pyplot as plt

from data.fbm import fractional_brownian_motion
from data.rough_vol import rough_volatility



def test_rough_vol():
    T = 1.0
    N = 500
    n_paths = 50

    v0 = 0.04        # initial variance (20% vol)
    eta = 1.0        # vol of vol
    hurst_values = [0.5, 0.3, 0.1]

    fig, axes = plt.subplots(len(hurst_values), 1, figsize=(10, 8))

    for i, H in enumerate(hurst_values):
        fbm = fractional_brownian_motion(
            H=H,
            T=T,
            N=N,
            n_paths=n_paths,
            seed=123,
        )

        v = rough_volatility(
            fbm_paths=fbm,
            v0=v0,
            eta=eta,
            T=T,
            H=H,
        )

        t = np.linspace(0, T, N + 1)

        for k in range(5):
            axes[i].plot(t, v[k], lw=1)

        axes[i].set_title(f"Rough volatility paths (H = {H})")
        axes[i].set_ylabel("v(t)")
        axes[i].set_ylim(bottom=0)

        print(
            f"H = {H} | min(v) = {v.min():.4e}, "
            f"mean(v_T) = {v[:, -1].mean():.4f}"
        )

    axes[-1].set_xlabel("t")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_rough_vol()
