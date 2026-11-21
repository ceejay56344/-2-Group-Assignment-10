"""06 Cohort Projection: self-contained section of HASTS206 Assignment 2.
Run from this project folder: python assignment_10_parts/06_cohort_projection.py
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
P20 = dist.set_index("district_id").loc[PD, "population_2020"]       # opening count 1 Jan 2020 (districts.csv)
P25 = agesex[agesex.district_id == PD].population_2025.sum()          # opening count 1 Jan 2025 (sum of age-sex baseline)
T = 5
arith_rate = (P25 - P20) / T                      # persons per year
geo_rate = (P25 / P20) ** (1 / T) - 1             # proportional growth per year
P30_arith = P25 + arith_rate * T
P30_geo = P25 * (1 + geo_rate) ** T
growth = pd.DataFrame({"Opening 1 Jan 2020": [P20, P20], "Opening 1 Jan 2025": [P25, P25],
                       "Fitted rate per year": [f"{arith_rate:,.1f} persons", f"{geo_rate*100:.3f}%"],
                       "Projected 1 Jan 2030": [P30_arith, P30_geo]}, index=["Arithmetic", "Geometric"])
growth
# ### 2(b) Five-year cohort-component projection

def project(did, row, fert_mult=None):
    """Advance 1 Jan 2025 age-sex counts to 1 Jan 2030. Timing: survival and births use the 2025 schedules for the full
    five years; net migration (5 x 2025 flow x multiplier) is added at the END of the period at the age group of arrival."""
    x = agesex[agesex.district_id == did]
    pv = lambda c: x.pivot(index="age_start", columns="sex", values=c)
    pop, sv, arr, dep = pv("population_2025"), pv("survival_5yr"), pv("arrivals_2025"), pv("departures_2025")
    fm = row.fertility_multiplier if fert_mult is None else fert_mult
    surv = pop * sv                                    # survivors of each starting cohort
    adv = surv.shift(1).fillna(0.0)                    # age group a -> a+5
    adv.loc[85] = surv.loc[80] + surv.loc[85]          # 85+ = survivors of 80-84 plus survivors already 85+
    births = 5 * x.births_to_women_2025.sum() * fm
    nb = pd.Series({"F": births * 0.49 * row.newborn_survival_f, "M": births * 0.51 * row.newborn_survival_m})
    adv.loc[0] = nb                                    # surviving births enter ages 0-4
    mig = 5 * (arr * row.arrivals_multiplier - dep * row.departures_multiplier)
    out = adv + mig                                    # end-of-period migration by destination age and sex
    chk = {"start": pop.values.sum(), "survivors": surv.values.sum(), "births": births, "surviving_births": nb.sum(),
           "net_migration": mig.values.sum(), "end": out.values.sum(),
           "deaths_in_period": pop.values.sum() - surv.values.sum() + births - nb.sum()}
    return out, chk

proj, checks = {}, []
for _, row in scen.iterrows():
    out, chk = project(PD, row)
    proj[row.scenario] = out
    chk["scenario"] = row.scenario
    chk["accounting_gap"] = chk["end"] - (chk["survivors"] + chk["surviving_births"] + chk["net_migration"])
    checks.append(chk)
acct = pd.DataFrame(checks).set_index("scenario")
acct

def summary(out):
    tot = out.values.sum()
    u15 = out.loc[[0, 5, 10]].values.sum(); o65 = out.loc[65:].values.sum()
    return {"Total 1 Jan 2030": tot, "Ages 0-4": out.loc[0].sum(), "Share under 15 (%)": u15 / tot * 100, "Share 65+ (%)": o65 / tot * 100}
comp = pd.DataFrame({s: summary(o) for s, o in proj.items()}).T
comp.loc["Arithmetic (total only)"] = [P30_arith, np.nan, np.nan, np.nan]
comp.loc["Geometric (total only)"] = [P30_geo, np.nan, np.nan, np.nan]
comp["Change vs 2025 (%)"] = (comp["Total 1 Jan 2030"] / P25 - 1) * 100
print(f"Accounting check: max |end - (survivors + surviving births + net migration)| = {acct.accounting_gap.abs().max():.2e}")
print(f"2025 shares for reference: under 15 = {agesex[(agesex.district_id==PD)&(agesex.age_start<15)].population_2025.sum()/P25*100:.1f}%, 65+ = {agesex[(agesex.district_id==PD)&(agesex.age_start>=65)].population_2025.sum()/P25*100:.1f}%")
comp

# **Comparison.** Base projects 180,173 people on 1 January 2030 (+4.5%), Low 173,772 (+0.8%) and High 186,644 (+8.2%). The arithmetic (188,281) and geometric (189,882) extrapolations are higher than every scenario. The reason is that they carry forward the 2020 to 2025 pace of about 1.9% a year, while actual growth in 2025 was only 0.8% and the 2025 net migration is negative. The cohort-component Base also shows an ageing shift only the scenarios can capture: the share under 15 is 29.5% (2025: 29.3%) and the share aged 65+ is 13.6% (2025: 14.6%); across scenarios these range from 28.5% to 30.4% and 13.5% to 13.7%.

fig, ax = plt.subplots(figsize=(8, 3.8))
labels = list(comp.index); vals = comp["Total 1 Jan 2030"].values
ax.bar(labels, vals, color=[NAVY, NAVY, NAVY, AMBER, AMBER])
ax.axhline(P25, color="grey", ls="--"); ax.text(4.4, P25, " 2025", va="bottom", color="grey")
ax.set_ylim(P25 * 0.95, vals.max() * 1.03); ax.set_ylabel("Persons"); plt.xticks(rotation=20, ha="right")
ax.set_title(f"{PD}: projected population 1 Jan 2030 (hypothetical data)"); plt.tight_layout()

pd.concat({s: o.stack().rename("population_2030") for s, o in proj.items()}, names=["scenario", "age_start", "sex"]).reset_index().to_csv(OUT / "projection_age_sex.csv", index=False)
comp.reset_index(names="model").to_csv(OUT / "projection_totals.csv", index=False)

# Save all figures created by this section.
FIG = Path("figures")
FIG.mkdir(exist_ok=True)
for number in plt.get_fignums():
    plt.figure(number).savefig(FIG / f"06_cohort_projection_{number}.png", bbox_inches="tight")
plt.close("all")
print("Finished: 06 Cohort Projection")
