import numpy as np
import tensorflow as tf


class Equation(object):
    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None

    def sample(self, num_sample):
        raise NotImplementedError

    def f_tf(self, t, x, y, z):
        raise NotImplementedError

    def g_tf(self, t, x):
        raise NotImplementedError


# ============================================================
# AMERICAN PUT — BLACK–SCHOLES (THIS IS NEW AND CORRECT)
# ============================================================

# class AmericanPutBS(Equation):
#     def __init__(self, eqn_config):
#         super().__init__(eqn_config)
#         self.s0 = eqn_config.s0
#         self.strike = eqn_config.strike
#         self.sigma = eqn_config.sigma
#         self.rate = eqn_config.rate
#         self.x_init = np.ones(self.dim) * self.s0

#     def sample(self, num_sample):
#         dw = np.random.normal(
#             size=[num_sample, self.dim, self.num_time_interval]
#         ) * self.sqrt_delta_t

#         x = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
#         x[:, :, 0] = self.x_init

#         for t in range(self.num_time_interval):
#             x[:, :, t + 1] = x[:, :, t] * np.exp(
#                 (self.rate - 0.5 * self.sigma ** 2) * self.delta_t
#                 + self.sigma * dw[:, :, t]
#             )

#         return dw, x

#     def f_tf(self, t, x, y, z):
#         return -self.rate * y

#     def g_tf(self, t, x):
#         return tf.maximum(self.strike - x, 0.0)

class AmericanPutRoughBS(Equation):
    def __init__(self, eqn_config):
        super().__init__(eqn_config)

        self.s0 = eqn_config.s0
        self.strike = eqn_config.strike
        self.rate = eqn_config.rate

        self.v0 = eqn_config.v0
        self.eta = eqn_config.eta
        self.hurst = eqn_config.hurst

        # augmented state dimension: [S_t, v_t]
        self.dim = 2

        self.x_init = np.array([self.s0, self.v0])

    def sample(self, num_sample):
        """
        Returns:
            dw : Brownian increments for asset
            x  : state paths, shape (num_sample, 2, N+1)
        """
        from data.fbm import fractional_brownian_motion
        from data.rough_vol import rough_volatility
        from data.rough_asset import simulate_rough_vol_asset

        # --- time grid ---
        N = self.num_time_interval
        dt = self.delta_t

        # --- fractional Brownian motion ---
        fbm = fractional_brownian_motion(
            H=self.hurst,
            T=self.total_time,
            N=N,
            n_paths=num_sample,
        )

        # --- rough volatility ---
        v_paths = rough_volatility(
            fbm_paths=fbm,
            v0=self.v0,
            eta=self.eta,
            T=self.total_time,
            H=self.hurst,
        )

        # --- Brownian motion for asset ---
        dw = np.random.normal(
            size=(num_sample, N)
        ) * np.sqrt(dt)

        # --- asset price ---
        S_paths = simulate_rough_vol_asset(
            S0=self.s0,
            r=self.rate,
            v_paths=v_paths,
            dt=dt,
            dW=dw,
        )

        # --- build augmented state ---
        x = np.zeros((num_sample, 2, N + 1))
        x[:, 0, :] = S_paths
        x[:, 1, :] = v_paths

        return dw, x

    def f_tf(self, t, x, y, z):
        # discounting only
        return -self.rate * y

    def g_tf(self, t, x):
        # x[..., 0] = S_t
        S = x[..., 0:1]
        return tf.maximum(self.strike - S, 0.0)


# ============================================================
# ORIGINAL EQUATIONS (UNCHANGED)
# ============================================================

class HJBLQ(Equation):
    def __init__(self, eqn_config):
        super().__init__(eqn_config)
        self.x_init = np.zeros(self.dim)
        self.sigma = np.sqrt(2.0)
        self.lambd = 1.0

    def sample(self, num_sample):
        dw = np.random.normal(
            size=[num_sample, self.dim, self.num_time_interval]
        ) * self.sqrt_delta_t

        x = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x[:, :, 0] = self.x_init

        for t in range(self.num_time_interval):
            x[:, :, t + 1] = x[:, :, t] + self.sigma * dw[:, :, t]

        return dw, x

    def f_tf(self, t, x, y, z):
        return -self.lambd * tf.reduce_sum(tf.square(z), 1, keepdims=True) / 2

    def g_tf(self, t, x):
        return tf.math.log((1 + tf.reduce_sum(tf.square(x), 1, keepdims=True)) / 2)
