import numpy as np


def rough_volatility(
    fbm_paths: np.ndarray,
    v0: float,
    eta: float,
    T: float,
    H: float,
):
    """
    Construct rough volatility paths from fractional Brownian motion.

    Model:
        log v_t = log v0 + eta * B_t^H - 0.5 * eta^2 * t^{2H}

    Parameters
    ----------
    fbm_paths : np.ndarray
        Shape (n_paths, N+1), fractional Brownian motion paths
    v0 : float
        Initial variance level
    eta : float
        Volatility of volatility
    T : float
        Time horizon
    H : float
        Hurst parameter

    Returns
    -------
    np.ndarray
        Shape (n_paths, N+1), volatility paths v_t > 0
    """

    n_paths, n_steps = fbm_paths.shape
    N = n_steps - 1
    t = np.linspace(0.0, T, N + 1)

    drift = -0.5 * (eta ** 2) * (t ** (2 * H))
    log_v = np.log(v0) + eta * fbm_paths + drift

    v_paths = np.exp(log_v)

    return v_paths
