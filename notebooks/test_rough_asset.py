import numpy as np
import matplotlib.pyplot as plt

from data.fbm import fractional_brownian_motion
from data.rough_vol import rough_volatility
from data.rough_asset import simulate_rough_vol_asset


def test_rough_asset():
    T = 1.0
    N = 500
    dt = T / N
    n_paths = 20

    S0 = 100.0
    r = 0.05
    v0 = 0.04
    eta = 1.0
    H = 0.1

    fbm = fractional_brownian_motion(
        H=H,
        T=T,
        N=N,
        n_paths=n_paths,
        seed=42,
    )

    v = rough_volatility(
        fbm_paths=fbm,
        v0=v0,
        eta=eta,
        T=T,
        H=H,
    )

    dW = np.random.normal(
        size=(n_paths, N)
    ) * np.sqrt(dt)

    S = simulate_rough_vol_asset(
        S0=S0,
        r=r,
        v_paths=v,
        dt=dt,
        dW=dW,
    )

    t = np.linspace(0, T, N + 1)

    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.plot(t, S[i], lw=1)

    plt.title("Asset price paths under rough volatility (H = 0.1)")
    plt.xlabel("t")
    plt.ylabel("S(t)")
    plt.grid(True)
    plt.show()

    print(
        f"min(S) = {S.min():.4f}, "
        f"mean(S_T) = {S[:, -1].mean():.2f}"
    )


if __name__ == "__main__":
    test_rough_asset()
