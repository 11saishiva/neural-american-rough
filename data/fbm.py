import numpy as np


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
