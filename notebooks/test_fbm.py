import numpy as np
import matplotlib.pyplot as plt

# from data.fbm import fractional_brownian_motion


def fractional_brownian_motion(
    H: float,
    T: float,
    N: int,
    n_paths: int,
    seed: int | None = None,
):
    """
    Generate fractional Brownian motion paths using covariance construction.

    Parameters
    ----------
    H : float
        Hurst parameter (0 < H < 1)
    T : float
        Time horizon
    N : int
        Number of time steps
    n_paths : int
        Number of sample paths
    seed : int or None
        Random seed

    Returns
    -------
    np.ndarray
        Shape (n_paths, N+1) containing fBM paths
    """

    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    t = np.linspace(0, T, N + 1)

    # --- covariance matrix ---
    cov = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            cov[i, j] = 0.5 * (
                t[i] ** (2 * H)
                + t[j] ** (2 * H)
                - abs(t[i] - t[j]) ** (2 * H)
            )

    # --- Cholesky decomposition ---
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(N + 1))

    # --- generate paths ---
    Z = np.random.normal(size=(n_paths, N + 1))
    fbm_paths = Z @ L.T

    return fbm_paths


def test_fbm():
    T = 1.0
    N = 500
    n_paths = 200
    hurst_values = [0.5, 0.3, 0.1]

    fig, axes = plt.subplots(len(hurst_values), 3, figsize=(15, 10))

    for i, H in enumerate(hurst_values):
        fbm = fractional_brownian_motion(
            H=H,
            T=T,
            N=N,
            n_paths=n_paths,
            seed=42,
        )

        t = np.linspace(0, T, N + 1)

        # --------------------------------------------------
        # 1️⃣ Sample paths (roughness visualization)
        # --------------------------------------------------
        for k in range(5):
            axes[i, 0].plot(t, fbm[k], lw=1)

        axes[i, 0].set_title(f"Sample paths (H = {H})")
        axes[i, 0].set_xlabel("t")
        axes[i, 0].set_ylabel("B_H(t)")

        # --------------------------------------------------
        # 2️⃣ Variance scaling: Var[B(t)] ~ t^(2H)
        # --------------------------------------------------
        var_t = np.var(fbm, axis=0)

        # avoid t=0
        log_t = np.log(t[1:])
        log_var = np.log(var_t[1:])

        coeff = np.polyfit(log_t, log_var, 1)
        slope = coeff[0]

        axes[i, 1].plot(log_t, log_var, label=f"slope ≈ {slope:.2f}")
        axes[i, 1].plot(
            log_t,
            np.polyval(coeff, log_t),
            linestyle="--",
        )
        axes[i, 1].set_title(f"Variance scaling (H = {H})")
        axes[i, 1].set_xlabel("log(t)")
        axes[i, 1].set_ylabel("log(Var)")
        axes[i, 1].legend()

        print(f"H = {H} | estimated slope ≈ {slope:.3f} (expected {2*H:.3f})")

        # --------------------------------------------------
        # 3️⃣ Increment correlation (memory)
        # --------------------------------------------------
        increments = np.diff(fbm, axis=1)
        corr = np.corrcoef(increments[:, :-1].ravel(),
                           increments[:, 1:].ravel())[0, 1]

        axes[i, 2].hist(
            increments[:, 1:].ravel(),
            bins=50,
            density=True,
            alpha=0.7,
        )
        axes[i, 2].set_title(
            f"Increment distribution (H={H})\n"
            f"lag-1 corr ≈ {corr:.3f}"
        )
        axes[i, 2].set_xlabel("ΔB")
        axes[i, 2].set_ylabel("Density")

        print(f"H = {H} | increment lag-1 correlation ≈ {corr:.3f}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_fbm()
