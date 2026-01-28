# import numpy as np


# def american_put_binomial(
#     S0: float,
#     K: float,
#     r: float,
#     sigma: float,
#     T: float,
#     N: int,
# ):
#     """
#     Cox–Ross–Rubinstein binomial tree for an American put option.

#     Parameters
#     ----------
#     S0 : float
#         Initial stock price
#     K : float
#         Strike price
#     r : float
#         Risk-free interest rate
#     sigma : float
#         Volatility
#     T : float
#         Time to maturity
#     N : int
#         Number of time steps

#     Returns
#     -------
#     float
#         American put option price at t=0
#     """

#     dt = T / N
#     u = np.exp(sigma * np.sqrt(dt))
#     d = 1.0 / u

#     disc = np.exp(-r * dt)
#     p = (np.exp(r * dt) - d) / (u - d)

#     # --- terminal stock prices ---
#     ST = np.array([S0 * (u ** j) * (d ** (N - j)) for j in range(N + 1)])

#     # --- terminal payoff ---
#     V = np.maximum(K - ST, 0.0)

#     # --- backward induction ---
#     for i in range(N - 1, -1, -1):
#         ST = ST[:-1] / u
#         continuation = disc * (p * V[1:] + (1.0 - p) * V[:-1])
#         exercise = np.maximum(K - ST, 0.0)
#         V = np.maximum(exercise, continuation)

#     return float(V[0])


# if __name__ == "__main__":
#     # Quick sanity check (not used in main pipeline)
#     price = american_put_binomial(
#         S0=100.0,
#         K=100.0,
#         r=0.05,
#         sigma=0.2,
#         T=1.0,
#         N=50,
#     )
#     print(f"American put (binomial): {price:.4f}")

import numpy as np


def american_put_binomial(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int,
):
    """
    Cox–Ross–Rubinstein binomial tree for an American put option.
    """

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u

    disc = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)

    # --- terminal payoff ---
    V = np.zeros(N + 1)
    for j in range(N + 1):
        ST = S0 * (u ** j) * (d ** (N - j))
        V[j] = max(K - ST, 0.0)

    # --- backward induction ---
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            ST = S0 * (u ** j) * (d ** (i - j))
            continuation = disc * (p * V[j + 1] + (1.0 - p) * V[j])
            exercise = max(K - ST, 0.0)
            V[j] = max(exercise, continuation)

    return float(V[0])


if __name__ == "__main__":
    price = american_put_binomial(
        S0=100.0,
        K=100.0,
        r=0.05,
        sigma=0.2,
        T=1.0,
        N=200,
    )
    print(f"American put (binomial): {price:.4f}")
