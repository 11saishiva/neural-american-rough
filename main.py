import json
import os
import logging
import numpy as np
import tensorflow as tf

from absl import app, flags
from absl import logging as absl_logging

import equation as eqn
from solver import BSDESolver

flags.DEFINE_string(
    "config_path",
    "configs/american_put_rough_bs.json",
    "Path to config file",
)

flags.DEFINE_string("exp_name", "american_put_test", "Experiment name")
FLAGS = flags.FLAGS
FLAGS.log_dir = "./logs"


def main(argv):
    del argv
    tf.keras.backend.clear_session()

    with open(FLAGS.config_path) as f:
        config_dict = json.load(f)

    class DictObj:
        def __init__(self, d):
            self.__dict__.update(d)

    class Config:
        def __init__(self, d):
            self.eqn_config = DictObj(d["eqn_config"])
            self.net_config = DictObj(d["net_config"])
            self.raw = d

        def to_dict(self):
            return self.raw

    config = Config(config_dict)
    tf.keras.backend.set_floatx(config.net_config.dtype)

    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    solver = BSDESolver(config, bsde)

    os.makedirs(FLAGS.log_dir, exist_ok=True)
    prefix = os.path.join(FLAGS.log_dir, FLAGS.exp_name)

    with open(prefix + "_config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    absl_logging.get_absl_handler().setFormatter(
        logging.Formatter("%(levelname)-6s %(message)s")
    )
    absl_logging.set_verbosity("info")

    logging.info("Begin to solve %s", config.eqn_config.eqn_name)
    history = solver.train()

    np.savetxt(
        prefix + "_training.csv",
        history,
        delimiter=",",
        header="step,loss,Y0,time",
        comments="",
    )


if __name__ == "__main__":
    app.run(main)
