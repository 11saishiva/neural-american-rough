# VolterraEx - Neural BSDE Pricing of American Options under Rough Volatility

This repository implements a **Neural Backward Stochastic Differential Equation (BSDE)** framework for pricing **American-style options under rough volatility**. The project combines modern deep learning techniques with stochastic analysis to solve high-dimensional optimal stopping problems that are analytically intractable using classical methods.

The core contribution is a **neural reflected BSDE solver** capable of handling **non-Markovian rough volatility dynamics**, driven by fractional Brownian motion (fBM).

---

## Motivation

American options allow early exercise, which leads to an optimal stopping problem. When volatility follows a **rough (fractional) process**, the problem becomes:

- **Non-Markovian**
- **Path-dependent**
- **High-dimensional**

Classical methods (trees, PDE grids) struggle or break down in this setting.

Neural BSDE methods:
- Avoid state-space discretization
- Scale better with dimension
- Naturally integrate Monte Carlo simulation

This project demonstrates that **neural BSDEs can price American options under rough volatility** and reveals how **volatility roughness impacts early-exercise value**.

---

## Method Overview

### Rough Volatility Model
- Volatility driven by fractional Brownian motion with Hurst parameter \( H < 0.5 \)
- Captures empirical features of financial volatility (roughness, long memory)

### American Option Formulation
- Pricing formulated as a **reflected BSDE**
- Reflection enforces the early-exercise constraint
- State augmented to include both asset price and volatility

### Neural BSDE Solver
- One feedforward neural network per time step
- Networks approximate the BSDE control process
- Training minimizes terminal payoff mismatch
- Implemented in TensorFlow

---

## Repository Structure

```

neural-american-rough/
│
├── main.py                  # Entry point for training
├── solver.py                # Neural BSDE solver
├── equation.py              # Rough volatility + payoff definitions
│
├── data/
│   ├── fbm.py               # Fractional Brownian motion simulation
│   ├── rough_vol.py         # Rough volatility construction
│   └── rough_asset.py       # Asset price simulation
│
├── benchmarks/
│   └── binomial_tree.py     # American put benchmark (Black–Scholes)
│
├── notebooks/
│   └── test_fbm.py          # fBM validation and diagnostics
│
├── configs/
│   └── american_put_rough_bs.json
│
├── logs/
│   └── *_training.csv       # Training outputs
│
├── analysis/
│   └── figures/             # Generated plots
│
└── README.md

````

---

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy tensorflow matplotlib absl-py
````

---

## Running the Model

Train the neural BSDE solver:

```bash
python main.py
```

Configuration is controlled via JSON files in `configs/`.

---

## Numerical Results

### Dependence on Volatility Roughness

The initial American put value ( Y_0 ) exhibits a **non-monotonic dependence on the Hurst parameter**:

| Hurst ( H ) | ( Y_0 ) |
| ----------- | ------- |
| 0.1         | ~4.94   |
| 0.3         | ~7.64   |
| 0.5         | ~4.16   |

This suggests a **trade-off between volatility persistence and irregularity** in early-exercise decisions.

### Convergence Behavior

* Stable convergence across all H values
* Higher roughness leads to slower convergence
* Training loss decreases consistently

Plots are saved automatically in:

```
analysis/figures/
```

---

## Benchmarks

A binomial tree implementation is provided for the classical Black–Scholes case:

```bash
python benchmarks/binomial_tree.py
```

Used for sanity checks and validation in the Markovian limit.

---

## Key Findings

* Neural BSDEs can handle **rough, non-Markovian volatility**
* American option values are sensitive to volatility roughness
* Rougher volatility does **not** monotonically increase option value
* The method is numerically stable and scalable

---

## Reproducibility

All experiments:

* Use fixed hyperparameters across H values
* Log training loss and value estimates to CSV
* Can be reproduced by rerunning `main.py`

---


## Acknowledgement
This work builds upon the Deep BSDE framework introduced by Han, Jentzen, and E (2018).
The original reference implementation by Frank Han was used as a baseline and substantially extended to support American options and rough volatility dynamics.
