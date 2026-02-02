import logging
import time
import numpy as np
import tensorflow as tf

DELTA_CLIP = 50.0
LAMBDA_PENALTY = 50.0


class BSDESolver(object):
    def __init__(self, config, bsde):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde

        self.model = NonsharedModel(config, bsde)
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.net_config.lr_boundaries, self.net_config.lr_values
        )
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, epsilon=1e-8
        )

    def train(self):
        start_time = time.time()
        history = []
        valid_data = self.bsde.sample(self.net_config.valid_size)

        for step in range(self.net_config.num_iterations + 1):
            if step % self.net_config.logging_frequency == 0:
                loss = self.loss_fn(valid_data, training=False).numpy()
                y0 = self.model.y_init.numpy()[0]
                elapsed = time.time() - start_time
                history.append([step, loss, y0, elapsed])

                if self.net_config.verbose:
                    logging.info(
                        "step: %5u, loss: %.4e, Y0: %.4e, time: %3u"
                        % (step, loss, y0, elapsed)
                    )

            self.train_step(self.bsde.sample(self.net_config.batch_size))

        return np.array(history)

    def loss_fn(self, inputs, training):
        dw, x = inputs
        y_terminal = self.model(inputs, training=training)

        payoff = self.bsde.g_tf(self.bsde.total_time, x[:, :, -1])
        delta = y_terminal - payoff

        base_loss = tf.reduce_mean(
            tf.where(
                tf.abs(delta) < DELTA_CLIP,
                tf.square(delta),
                2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2,
            )
        )

        # American early-exercise penalty
        penalty = tf.nn.relu(payoff - y_terminal)
        penalty_loss = LAMBDA_PENALTY * tf.reduce_mean(penalty)

        return base_loss + penalty_loss

    @tf.function
    def train_step(self, train_data):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(train_data, training=True)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


class NonsharedModel(tf.keras.Model):
    def __init__(self, config, bsde):
        super().__init__()
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde

        self.y_init = self.add_weight(
            name="y_init",
            shape=[1],
            initializer=tf.random_uniform_initializer(
                self.net_config.y_init_range[0],
                self.net_config.y_init_range[1],
            ),
        )

        self.z_init = self.add_weight(
            name="z_init",
            shape=[1, self.eqn_config.dim],
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
        )

        self.subnet = [
            FeedForwardSubNet(config)
            for _ in range(self.bsde.num_time_interval - 1)
        ]
    def call(self, inputs, training=None):
        dw, x = inputs
        batch = tf.shape(dw)[0]

        ones = tf.ones([batch, 1], dtype=self.net_config.dtype)

        # Initial conditions
        y = ones * self.y_init
        z = tf.matmul(ones, self.z_init)

        # Time stepping (N-1 steps)
        for t in range(self.bsde.num_time_interval - 1):
            y = (
                y
                - self.bsde.delta_t
                * self.bsde.f_tf(
                    t * self.bsde.delta_t,
                    x[:, :, t],
                    y,
                    z,
                )
                + z[:, 0:1] * dw[:, t:t+1]
            )

            # Z-network exists only for first N-1 steps
            z = self.subnet[t](x[:, :, t + 1], training=training)

        # Final drift step (NO new Z, NO noise)
        y = (
                y
                - self.bsde.delta_t
                * self.bsde.f_tf(
                    t * self.bsde.delta_t,
                    x[:, :, t],
                    y,
                    z,
                )
                + z[:, 0:1] * dw[:, t:t+1]
            )

            # --- AMERICAN EARLY EXERCISE PROJECTION ---
        payoff = self.bsde.g_tf(t * self.bsde.delta_t, x[:, :, t])
        y = tf.maximum(y, payoff)

        return y


class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        dim = config.eqn_config.dim
        num_hiddens = config.net_config.num_hiddens

        self.bn_layers = [
            tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-6)
            for _ in range(len(num_hiddens) + 2)
        ]

        self.dense_layers = [
            tf.keras.layers.Dense(h, use_bias=False) for h in num_hiddens
        ]
        self.dense_layers.append(tf.keras.layers.Dense(dim))

    def call(self, x, training=None):
        x = self.bn_layers[0](x, training=training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i + 1](x, training=training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x, training=training)
        return x
