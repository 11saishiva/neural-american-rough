import numpy as np


def simulate_rough_vol_asset(
    S0: float,
    r: float,
    v_paths: np.ndarray,
    dt: float,
    dW: np.ndarray,
):
    """
    Simulate asset price paths under rough volatility.

    Model:
        S_{t+dt} = S_t * exp(
            (r - 0.5 * v_t) * dt + sqrt(v_t) * dW_t
        )

    Parameters
    ----------
    S0 : float
        Initial asset price
    r : float
        Risk-free rate
    v_paths : np.ndarray
        Shape (n_paths, N+1), variance paths
    dt : float
        Time step size
    dW : np.ndarray
        Shape (n_paths, N), Brownian increments

    Returns
    -------
    np.ndarray
        Shape (n_paths, N+1), asset price paths
    """

    n_paths, N_plus_1 = v_paths.shape
    N = N_plus_1 - 1

    S = np.zeros((n_paths, N + 1))
    S[:, 0] = S0

    for t in range(N):
        vt = np.maximum(v_paths[:, t], 0.0)  # safety clamp
        S[:, t + 1] = S[:, t] * np.exp(
            (r - 0.5 * vt) * dt
            + np.sqrt(vt) * dW[:, t]
        )

    return S
