"""04 Life Table: self-contained section of HASTS206 Assignment 2.
Run from this project folder: python assignment_10_parts/04_life_table.py
The supplied CSVs must be in data/raw/.
"""
"""HASTS206 Assignment 2 analysis. Run from the repository root: python analysis.py"""
import matplotlib
matplotlib.use("Agg")

# # HASTS206 Assignment 2: Applied Demographic Analysis
# **All figures are hypothetical (synthetic data supplied by the lecturer).** Python does every calculation; the original CSVs in `data/raw/` are never edited. Cleaning decisions are recorded in code and in the data-quality log.

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import expm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
from sklearn.impute import KNNImputer

RAW, OUT = Path("data/raw"), Path("outputs")
OUT.mkdir(exist_ok=True)
SEED = 42
NAVY, AMBER = "#1B365D", "#D97724"
pd.options.display.float_format = lambda v: f"{v:,.0f}" if abs(v - round(v)) < 1e-9 else f"{v:,.3f}"
plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False})

dist = pd.read_csv(RAW / "districts.csv")
agesex = pd.read_csv(RAW / "age_sex.csv")
lt_in = pd.read_csv(RAW / "life_table_inputs.csv")
scen = pd.read_csv(RAW / "projection_scenarios.csv")
ins = pd.read_csv(RAW / "insured_transitions.csv")

# ### 1(c) Abridged life table

lt = lt_in.copy()
n, ax_ = lt.interval_years, lt.ax
lt["mx"] = lt.deaths / lt.person_years
lt["qx"] = n * lt.mx / (1 + (n - ax_) * lt.mx)
lt.loc[lt.index[-1], "qx"] = 1.0
l = [100000.0]
for q in lt.qx[:-1]: l.append(l[-1] * (1 - q))
lt["lx"] = l
lt["dx"] = lt.lx * lt.qx
Lx = [n[i] * lt.lx[i + 1] + ax_[i] * lt.dx[i] for i in range(len(lt) - 1)]
Lx.append(lt.lx.iloc[-1] / lt.mx.iloc[-1])           # L85+ = l85 / m85+
lt["Lx"] = Lx
lt["Tx"] = lt.Lx[::-1].cumsum()[::-1]
lt["ex"] = lt.Tx / lt.lx
life = lt[["age_start", "interval_years", "mx", "qx", "lx", "dx", "Lx", "Tx", "ex"]]
e0, e65 = lt.ex[lt.age_start == 0].item(), lt.ex[lt.age_start == 65].item()
print(f"e0 = {e0:.2f} years, e65 = {e65:.2f} years")
print(f"Check 1: sum of dx = {lt.dx.sum():,.1f} (must equal the radix 100,000)")
print(f"Check 2: l85 = {lt.lx.iloc[-1]:,.1f} equals radix minus deaths below 85 = {100000 - lt.dx[:-1].sum():,.1f}")
print(f"Check 3: T0 = {lt.Tx.iloc[0]:,.1f} equals the sum of Lx = {lt.Lx.sum():,.1f}")
life

# **Interpretation.** Under the supplied 2025 mortality schedule a newborn would live 73.1 years on average (e0) and a person who reaches 65 would live a further 18.1 years (e65, to about age 83). About 31.9% of the radix survives to age 85. Checks: the death column sums to the radix of 100,000, l85 equals the radix minus deaths below 85 (31,880), and T0 equals the sum of Lx.

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(lt.age_start, lt.lx, marker="o", color=NAVY); ax[0].set_title("Survivors lx (radix 100,000)"); ax[0].set_xlabel("Exact age (years)")
ax[1].plot(lt.age_start, lt.ex, marker="o", color=AMBER); ax[1].set_title("Life expectancy ex (years)"); ax[1].set_xlabel("Exact age (years)")
fig.suptitle("Abridged life table (hypothetical data)"); plt.tight_layout()

life.to_csv(OUT / "life_table.csv", index=False)

# Save all figures created by this section.
FIG = Path("figures")
FIG.mkdir(exist_ok=True)
for number in plt.get_fignums():
    plt.figure(number).savefig(FIG / f"04_life_table_{number}.png", bbox_inches="tight")
plt.close("all")
print("Finished: 04 Life Table")
