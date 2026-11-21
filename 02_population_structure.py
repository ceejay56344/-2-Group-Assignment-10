"""02 Population Structure: self-contained section of HASTS206 Assignment 2.
Run from this project folder: python assignment_10_parts/02_population_structure.py
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

g = agesex.groupby("district_id")
prof = dist.set_index("district_id").copy()
prof["pop_2025"] = g.population_2025.sum()
prof["share_u15"] = agesex[agesex.age_start < 15].groupby("district_id").population_2025.sum() / prof.pop_2025
prof["share_65p"] = agesex[agesex.age_start >= 65].groupby("district_id").population_2025.sum() / prof.pop_2025
prof["mid_pop_2025"] = (prof.pop_2025 + prof.population_2026) / 2
prof["nmr_2025"] = (prof.arrivals_2025 - prof.departures_2025) / prof.mid_pop_2025 * 1000
prof["cbr_2025"] = prof.births_2025 / prof.mid_pop_2025 * 1000
prof["cdr_2025"] = prof.deaths_2025 / prof.mid_pop_2025 * 1000

# Rule: among districts with at least 150,000 residents (comparable scale), take the youngest and oldest age structure.
big = prof[prof.pop_2025 >= 150000]
SEL = [big.share_u15.idxmax(), big.share_u15.idxmin()]
PD = SEL[0]    # projection district: births drive its growth, so fertility and migration scenarios both matter
print("Selected for comparison:", SEL, "| projection district:", PD)
big[["region", "pop_2025", "share_u15", "share_65p", "cbr_2025", "cdr_2025", "nmr_2025"]]
# ## 1. Population structure and mortality

AGES = sorted(agesex.age_start.unique())
LAB = [f"{a}-{a+4}" for a in AGES[:-1]] + ["85+"]
def pyr(did):
    return agesex[agesex.district_id == did].pivot(index="age_start", columns="sex", values="population_2025")
mx = np.ceil(max(pyr(d).values.max() for d in SEL) / 2000) * 2000
fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
for ax, did in zip(axes, SEL):
    x = pyr(did)
    ax.barh(range(len(AGES)), -x["M"], color=NAVY, label="Male")
    ax.barh(range(len(AGES)), x["F"], color=AMBER, label="Female")
    ax.set_xlim(-mx, mx); ax.set_yticks(range(len(AGES)), LAB)
    t = np.linspace(-mx, mx, 9); ax.set_xticks(t, [f"{abs(int(v)):,}" for v in t], fontsize=8)
    ax.set_title(f"{did}, 1 January 2025"); ax.set_xlabel("Persons per 5-year age group")
axes[0].set_ylabel("Age group (years)"); axes[0].legend(loc="upper left")
fig.suptitle("Population pyramids on a common scale (hypothetical data)")
plt.tight_layout()

def ratios(did):
    x = agesex[agesex.district_id == did]
    P = lambda lo, hi: x[(x.age_start >= lo) & (x.age_start <= hi)].population_2025.sum()
    m, f = x[x.sex == "M"].population_2025.sum(), x[x.sex == "F"].population_2025.sum()
    return {"Population 1 Jan 2025": m + f, "Males": m, "Females": f,
            "Sex ratio (males per 100 females)": m / f * 100,
            "Youth dependency ratio (0-14 per 100 aged 15-64)": P(0, 10) / P(15, 60) * 100,
            "Old-age dependency ratio (65+ per 100 aged 15-64)": P(65, 85) / P(15, 60) * 100}
ratio_tbl = pd.DataFrame({d: ratios(d) for d in SEL})
ratio_tbl

# **Interpreting the contrast.** D01 has a wide base and a narrow top: 52.2 children aged 0-14 for every 100 people aged 15-64, against 15.5 in D24. D24 is the opposite, with 44.2 people aged 65+ per 100 aged 15-64 against 26.1 in D01. D01 therefore carries a heavy child burden and D24 a heavy old-age burden. The sex ratio is lower in D01 (88.3 males per 100 females) than in D24 (94.7); the data do not show why.

# Save all figures created by this section.
FIG = Path("figures")
FIG.mkdir(exist_ok=True)
for number in plt.get_fignums():
    plt.figure(number).savefig(FIG / f"02_population_structure_{number}.png", bbox_inches="tight")
plt.close("all")
print("Finished: 02 Population Structure")
