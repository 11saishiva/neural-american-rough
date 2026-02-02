import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================
# CONFIG
# ==========================
LOG_DIR = "logs"
RUNS = {
    0.5: "american_put_H_0.5_training.csv",
    0.3: "american_put_H_0.3_training.csv",
    0.1: "american_put_H_0.1_training.csv",
}

OUTPUT_DIR = "analysis/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# LOAD DATA
# ==========================
data = {}

for H, fname in RUNS.items():
    path = os.path.join(LOG_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    data[H] = df

# ==========================
# 1. CONVERGENCE PLOT (Y0)
# ==========================
plt.figure(figsize=(8, 5))

for H, df in data.items():
    plt.plot(df["step"], df["Y0"], label=f"H = {H}")

plt.xlabel("Iteration")
plt.ylabel("Y₀ (Option Value)")
plt.title("Convergence of American Put Value under Rough Volatility")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Y0_convergence.png"), dpi=300)
plt.close()

# ==========================
# 2. LOSS DIAGNOSTIC
# ==========================
plt.figure(figsize=(8, 5))

for H, df in data.items():
    plt.plot(df["step"], df["loss"], label=f"H = {H}")

plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("Training Loss under Rough Volatility")
plt.yscale("log")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_convergence.png"), dpi=300)
plt.close()

# ==========================
# 3. SUMMARY TABLE
# ==========================
summary = []

for H, df in data.items():
    final_Y0 = df["Y0"].iloc[-1]
    summary.append({"H": H, "Y0": final_Y0})

summary_df = pd.DataFrame(summary).sort_values("H", ascending=False)

print("\n=== FINAL AMERICAN PUT VALUES ===")
print(summary_df.to_string(index=False))

summary_df.to_csv("analysis/summary_table.csv", index=False)

# ==========================
# 4. Y0 vs H (NON-MONOTONICITY)
# ==========================
plt.figure(figsize=(6, 4))

plt.plot(summary_df["H"], summary_df["Y0"], marker="o")
plt.xlabel("Hurst Parameter H")
plt.ylabel("Y₀ (Option Value)")
plt.title("Dependence of American Put Value on Volatility Roughness")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Y0_vs_H.png"), dpi=300)
plt.close()

print("\nPlots saved to:", OUTPUT_DIR)
