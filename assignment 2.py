"""HASTS206 Assignment 2: Applied Demographic Analysis. Complete analysis in one file.
All data are synthetic (hypothetical).

HOW TO RUN
1. Install the libraries once:
       pip install numpy pandas scipy scikit-learn matplotlib
2. Keep this file in a folder that contains  data/raw/  with the five CSVs:
       districts.csv, age_sex.csv, life_table_inputs.csv, projection_scenarios.csv, insured_transitions.csv
3. Run from that folder:   python hasts206_assignment2_full_analysis.py
Results are written to outputs/ (CSVs for Power BI) and figures/ (charts). The original CSVs are never edited.
"""
# ====================================================================================================
# LIBRARIES
# ====================================================================================================
import matplotlib
matplotlib.use("Agg")                                         # save charts to files; no window needed
import numpy as np                                            # arrays and numeric calculations
import pandas as pd                                           # tables and CSV files
import matplotlib.pyplot as plt                               # charts
from pathlib import Path                                      # file paths
from scipy.linalg import expm                                 # matrix exponential (Markov model)
from sklearn.cluster import KMeans                            # k-means clustering
from sklearn.preprocessing import StandardScaler              # standardising features
from sklearn.impute import KNNImputer                         # filling missing piped-water values
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score   # clustering diagnostics

# ====================================================================================================
# SETTINGS, DATA LOADING AND DISTRICT PROFILE TABLE
# ====================================================================================================
ROOT = Path(__file__).resolve().parent                        # the folder this file is in
RAW, OUT, FIG = ROOT / "data" / "raw", ROOT / "outputs", ROOT / "figures"
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
SEED = 42
NAVY, AMBER = "#1B365D", "#D97724"
pd.options.display.float_format = lambda v: f"{v:,.0f}" if abs(v - round(v)) < 1e-9 else f"{v:,.3f}"
plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False})

def fmt(x):                                                   # print tables in full
    return x.to_string() if hasattr(x, "to_string") else repr(x)

def save_fig(name):                                           # save every open chart to figures/ and close it
    for n in plt.get_fignums():
        plt.figure(n).savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(n)

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

# District selection rule: among districts with at least 150,000 residents (comparable scale), take the youngest and oldest age structure.
big = prof[prof.pop_2025 >= 150000]
SEL = [big.share_u15.idxmax(), big.share_u15.idxmin()]
PD = SEL[0]    # projection district: births drive its growth, so fertility and migration scenarios both matter
print("Selected for comparison:", SEL, "| projection district:", PD)




# ====================================================================================================
# SECTION 1: DATA-QUALITY LOG AND RECONCILIATION  (Before analysis)
# ====================================================================================================
# Before analysis: data-quality log

log = []
def add(check, result, decision):
    log.append({"check": check, "result": result, "decision": decision})

# 1. Row counts
for name, df in [("districts", dist), ("age_sex", agesex), ("life_table_inputs", lt_in),
                 ("projection_scenarios", scen), ("insured_transitions", ins)]:
    add(f"Row count: {name}", f"{len(df)} rows x {df.shape[1]} columns", "Matches expected structure (see below)")

# 2. Keys and join cardinality
add("Key unique: districts.district_id", f"unique = {dist.district_id.is_unique}, n = {dist.district_id.nunique()}", "Primary key")
add("Key unique: age_sex (district_id, age_start, sex)", f"duplicates = {agesex.duplicated(['district_id','age_start','sex']).sum()}", "Composite key valid")
add("Key unique: insured (district_id, origin, destination)", f"duplicates = {ins.duplicated(['district_id','origin','destination']).sum()}", "Composite key valid")
per = agesex.groupby("district_id").size()
add("Join districts -> age_sex", f"every district has {sorted(per.unique())} age-sex rows; unmatched ids = {len(set(agesex.district_id) ^ set(dist.district_id))}", "One-to-many (1 : 36), no orphans")
perI = ins.groupby("district_id").size()
add("Join districts -> insured_transitions", f"every district has {sorted(perI.unique())} rows; unmatched ids = {len(set(ins.district_id) ^ set(dist.district_id))}", "One-to-many (1 : 4), no orphans")

# 3. Coverage
add("Age-band and sex coverage", f"{agesex.age_start.nunique()} age bands (0-85+) x {agesex.sex.nunique()} sexes per district in all {agesex.district_id.nunique()} districts", "Complete")

# 4. Missing indicators
miss = dist.isna().sum(); miss = miss[miss > 0]
add("Missing values: districts", f"{dict(miss)} -> {list(dist.loc[dist.piped_water_pct.isna(), 'district_id'])}", "Keep rows; impute in Part 4 (kNN rule)")
add("Missing values: age_sex / insured / scenarios", f"{int(agesex.isna().sum().sum())} / {int(ins.isna().sum().sum())} / {int(scen.isna().sum().sum())}", "None")
add("Missing values: life_table_inputs", f"{int(lt_in.isna().sum().sum())} cells, all in the open 85+ row (interval_years, ax)", "Structural: the open interval uses L85+ = l85 / m85+")

# 5. Valid ranges
add("Range: survival_5yr", f"min {agesex.survival_5yr.min():.3f}, max {agesex.survival_5yr.max():.3f}", "Within (0, 1]")
cnt_cols = ["population_2025", "baseline_deaths_2025", "arrivals_2025", "departures_2025", "births_to_women_2025"]
add("Range: age_sex counts", f"min of counts = {int(agesex[cnt_cols].min().min())}", "No negatives")
add("Range: piped_water_pct", f"{dist.piped_water_pct.min()} to {dist.piped_water_pct.max()}", "Within 0-100")
add("Range: life table deaths <= person-years", str(bool((lt_in.deaths <= lt_in.person_years).all())), "Valid")
add("Range: scenario survival and multipliers", f"newborn survival {scen[['newborn_survival_f','newborn_survival_m']].min().min()}-{scen[['newborn_survival_f','newborn_survival_m']].max().max()}; multipliers > 0 = {bool((scen.drop(columns='scenario') > 0).all().all())}", "Valid")
add("Births only to women 15-49", f"ages with births > 0: {sorted(agesex.loc[agesex.births_to_women_2025 > 0, 'age_start'].unique())}; male births = {int(agesex.loc[agesex.sex=='M','births_to_women_2025'].sum())}", "Consistent")

# 6. Repeated fields in insured_transitions
rep = ins.groupby(["district_id", "origin"]).agg(e=("exposure_py", "nunique"), i=("initial_lives", "nunique"), r=("destination", "size"))
add("Repeated fields (exposure_py, initial_lives)", f"constant within each district-origin (max distinct = {rep.e.max()}, {rep.i.max()}) and repeated on {rep.r.max()} destination rows", "Count once per district-origin, never once per row")

# 7. Reconciliation
g = agesex.groupby("district_id")
rec = dist.set_index("district_id").join(pd.DataFrame({
    "pop25": g.population_2025.sum(), "b_as": g.births_to_women_2025.sum(), "d_as": g.baseline_deaths_2025.sum(),
    "a_as": g.arrivals_2025.sum(), "m_as": g.departures_2025.sum()}))
rec["recon_2026"] = rec.pop25 + rec.births_2025 - rec.deaths_2025 + rec.arrivals_2025 - rec.departures_2025
add("Births: district vs age_sex", f"max abs difference = {(rec.births_2025 - rec.b_as).abs().max()}", "Reconciled")
add("Arrivals / departures: district vs age_sex", f"max abs difference = {(rec.arrivals_2025 - rec.a_as).abs().max()} / {(rec.departures_2025 - rec.m_as).abs().max()}", "Reconciled")
add("Balancing equation: P2025 + B - D + in - out vs population_2026", f"max abs difference = {(rec.population_2026 - rec.recon_2026).abs().max()}", "Reconciled exactly, using district deaths")
dd = rec.deaths_2025 - rec.d_as
add("Deaths: district vs baseline age_sex", f"district total {rec.deaths_2025.sum():,} vs baseline {rec.d_as.sum():,}; district is higher in {(dd > 0).sum()} of 24 districts (gap {dd.min()} to {dd.max()})", "Different scope, see note below")
dq = pd.DataFrame(log)
dq.to_csv(OUT / "data_quality_log.csv", index=False)
print(fmt(dq))

# Scope difference between deaths. deaths_2025 in districts.csv counts every death in the district during
# 2025 and is the figure that closes the balancing equation. baseline_deaths_2025 in the age-sex file
# counts only deaths of residents present at baseline (1 January 2025), classified by age at baseline. As
# the supplied README states, district deaths additionally include deaths among 2025 births, so the
# district total is higher in every district. The district total is used in all balancing and rate
# calculations.


# ====================================================================================================
# SECTION 2: DISTRICT SELECTION, POPULATION PYRAMIDS AND RATIOS  (Question 1(a))
# ====================================================================================================
selection_table = big[["region", "pop_2025", "share_u15", "share_65p", "cbr_2025", "cdr_2025", "nmr_2025"]]
print(fmt(selection_table))
selection_table.to_csv(OUT / "district_selection.csv")

# Selecting districts

# Why these districts. Only districts of at least 150,000 residents were considered, so the pyramids and
# counts are on a comparable scale. Within that set, the highest and lowest under-15 shares give the two
# sharpest contrasts: a young, high-fertility district and an old district where deaths exceed births. The
# young district (D01) is also used for the projection because its growth comes from births, so the
# fertility and migration scenarios both change the result.

# 1. Population structure and mortality

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
save_fig("m02_population_pyramids")

def ratios(did):
    x = agesex[agesex.district_id == did]
    P = lambda lo, hi: x[(x.age_start >= lo) & (x.age_start <= hi)].population_2025.sum()
    m, f = x[x.sex == "M"].population_2025.sum(), x[x.sex == "F"].population_2025.sum()
    return {"Population 1 Jan 2025": m + f, "Males": m, "Females": f,
            "Sex ratio (males per 100 females)": m / f * 100,
            "Youth dependency ratio (0-14 per 100 aged 15-64)": P(0, 10) / P(15, 60) * 100,
            "Old-age dependency ratio (65+ per 100 aged 15-64)": P(65, 85) / P(15, 60) * 100}
ratio_tbl = pd.DataFrame({d: ratios(d) for d in SEL})
print(fmt(ratio_tbl))

# Interpretation for the report (check against your own output): Interpreting the contrast. D01 has a wide
# base and a narrow top: 52.2 children aged 0-14 for every 100 people aged 15-64, against 15.5 in D24. D24
# is the opposite, with 44.2 people aged 65+ per 100 aged 15-64 against 26.1 in D01. D01 therefore carries
# a heavy child burden and D24 a heavy old-age burden. The sex ratio is lower in D01 (88.3 males per 100
# females) than in D24 (94.7); the data do not show why.

ratio_tbl.to_csv(OUT / "ratios_selected_districts.csv")


# ====================================================================================================
# SECTION 3: POPULATION BALANCE AND DEMOGRAPHIC RATES  (Question 1(b))
# ====================================================================================================
# 1(b) Population balance and demographic rates

rows = {}
for d in SEL:
    r = prof.loc[d]
    p25 = agesex[agesex.district_id == d].population_2025.sum()
    recon = p25 + r.births_2025 - r.deaths_2025 + r.arrivals_2025 - r.departures_2025
    mid = (p25 + r.population_2026) / 2          # mean of the opening stocks, 1 Jan 2025 and 1 Jan 2026
    nat = r.births_2025 - r.deaths_2025; net = r.arrivals_2025 - r.departures_2025
    rows[d] = {"Baseline population 1 Jan 2025 (sum of age-sex)": p25, "Births 2025": r.births_2025, "Deaths 2025": r.deaths_2025,
               "Arrivals": r.arrivals_2025, "Departures": r.departures_2025,
               "Reconstructed 1 Jan 2026": recon, "Reported 1 Jan 2026": r.population_2026, "Difference": r.population_2026 - recon,
               "Mid-year exposure": mid, "CBR per 1,000": r.births_2025 / mid * 1000, "CDR per 1,000": r.deaths_2025 / mid * 1000,
               "Net migration rate per 1,000": net / mid * 1000, "Natural increase rate per 1,000": nat / mid * 1000,
               "Total growth rate per 1,000": (recon - p25) / mid * 1000}
bal = pd.DataFrame(rows)
print(fmt(bal))

# Interpretation for the report (check against your own output): Drivers of growth. The reconstructed 1
# January 2026 totals equal the reported values exactly in both districts (difference 0), so the balancing
# equation closes. D01 grew by 7.6 per 1,000: a natural increase of 10.2 per 1,000 (CBR 22.8, CDR 12.5)
# outweighed net out-migration of 2.7 per 1,000. D24 shrank by 9.7 per 1,000: its CDR (21.0) is far above
# its CBR (12.8), a natural decrease of 8.3 per 1,000, made worse by net out-migration of 1.4 per 1,000.
# D01 grows through births; D24 declines through deaths.

bal.to_csv(OUT / "balance_and_rates.csv")


# ====================================================================================================
# SECTION 4: ABRIDGED LIFE TABLE  (Question 1(c))
# ====================================================================================================
# 1(c) Abridged life table

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
print(fmt(life))

# Interpretation for the report (check against your own output): Interpretation. Under the supplied 2025
# mortality schedule a newborn would live 73.1 years on average (e0) and a person who reaches 65 would
# live a further 18.1 years (e65, to about age 83). About 31.9% of the radix survives to age 85. Checks:
# the death column sums to the radix of 100,000, l85 equals the radix minus deaths below 85 (31,880), and
# T0 equals the sum of Lx.

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(lt.age_start, lt.lx, marker="o", color=NAVY); ax[0].set_title("Survivors lx (radix 100,000)"); ax[0].set_xlabel("Exact age (years)")
ax[1].plot(lt.age_start, lt.ex, marker="o", color=AMBER); ax[1].set_title("Life expectancy ex (years)"); ax[1].set_xlabel("Exact age (years)")
fig.suptitle("Abridged life table (hypothetical data)"); plt.tight_layout()
save_fig("m04_life_table_plots")

life.to_csv(OUT / "life_table.csv", index=False)


# ====================================================================================================
# SECTION 5: COHORT-COMPONENT PROJECTION UNDER THREE SCENARIOS  (Question 2(b))
# ====================================================================================================
# 2(b) Five-year cohort-component projection

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
print(fmt(acct))

pd.concat({s: o.stack().rename("population_2030") for s, o in proj.items()}, names=["scenario", "age_start", "sex"]).reset_index().to_csv(OUT / "projection_age_sex.csv", index=False)

acct.to_csv(OUT / "projection_accounting_checks.csv")


# ====================================================================================================
# SECTION 6: GROWTH MODELS, COMPARISON WITH SCENARIOS AND FERTILITY SENSITIVITY  (Questions 2(a) and 2(c))
# ====================================================================================================
# 2(a) Arithmetic and geometric growth

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
print(fmt(growth))

# Estimates versus projections. The 2020 and 2025 opening counts and the fitted growth rates are estimates
# from the supplied (hypothetical) data. The 2030 totals are conditional projections: they hold only if
# the 2020 to 2025 growth pattern continues unchanged and say nothing about births, deaths, migration or
# age structure.

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
print(fmt(comp))

# Interpretation for the report (check against your own output): Comparison. Base projects 180,173 people
# on 1 January 2030 (+4.5%), Low 173,772 (+0.8%) and High 186,644 (+8.2%). The arithmetic (188,281) and
# geometric (189,882) extrapolations are higher than every scenario. The reason is that they carry forward
# the 2020 to 2025 pace of about 1.9% a year, while actual growth in 2025 was only 0.8% and the 2025 net
# migration is negative. The cohort-component Base also shows an ageing shift only the scenarios can
# capture: the share under 15 is 29.5% (2025: 29.3%) and the share aged 65+ is 13.6% (2025: 14.6%); across
# scenarios these range from 28.5% to 30.4% and 13.5% to 13.7%.

fig, ax = plt.subplots(figsize=(8, 3.8))
labels = list(comp.index); vals = comp["Total 1 Jan 2030"].values
ax.bar(labels, vals, color=[NAVY, NAVY, NAVY, AMBER, AMBER])
ax.axhline(P25, color="grey", ls="--"); ax.text(4.4, P25, " 2025", va="bottom", color="grey")
ax.set_ylim(P25 * 0.95, vals.max() * 1.03); ax.set_ylabel("Persons"); plt.xticks(rotation=20, ha="right")
ax.set_title(f"{PD}: projected population 1 Jan 2030 (hypothetical data)"); plt.tight_layout()
save_fig("m05_projected_totals")

# 2(c) Fertility sensitivity

base_row = scen[scen.scenario == "Base"].iloc[0]
high_f = scen.loc[scen.scenario == "High", "fertility_multiplier"].item()
b_out, _ = project(PD, base_row)
h_out, _ = project(PD, base_row, fert_mult=high_f)       # only the fertility multiplier changes
sens = pd.DataFrame({"Base": [b_out.loc[0].sum(), b_out.values.sum()],
                     f"Base with fertility x{high_f}": [h_out.loc[0].sum(), h_out.values.sum()]}, index=["Ages 0-4", "Total"])
sens["Change"] = sens.iloc[:, 1] - sens.iloc[:, 0]; sens["Change (%)"] = sens.Change / sens.iloc[:, 0] * 100
print(fmt(sens))

# Interpretation for the report (check against your own output): Sensitivity. Raising only the fertility
# multiplier from 1.00 to 1.12 lifts projected births from 19,695 to 22,058 and the 0-4 population from
# 18,831 to 21,133: 2,302 more children (+12.2%). The total rises by the same 2,302 (+1.3%), because only
# ages 0-4 change. *Planning implication:* about 2,300 extra under-fives by 2030 would need more early-
# childhood health and school places from the early 2030s. *Limitation:* births are five times the 2025
# births, scaled by the multiplier, and are not recalculated from the changing number of women in
# childbearing ages, so the result holds only for a five-year horizon.

comp.reset_index(names="model").to_csv(OUT / "projection_totals.csv", index=False)

growth.to_csv(OUT / "growth_models.csv")

sens.to_csv(OUT / "fertility_sensitivity.csv")


# ====================================================================================================
# SECTION 7: THREE-STATE MODEL: GENERATOR MATRIX, DIAGRAM, HOLDING TIMES  (Questions 3(a) and 3(b))
# ====================================================================================================
# 3(a) State diagram and generator matrix

# README repeated-field rule: exposure_py and initial_lives repeat on each destination row, so take one value per district-origin.
once = ins.drop_duplicates(["district_id", "origin"])
exposure = once.groupby("origin").exposure_py.sum()          # pooled person-years at risk in each origin state
initial = once.groupby("origin").initial_lives.sum()         # pooled initial counts
jumps = ins.groupby(["origin", "destination"]).transition_count.sum()
rate = {(o, d): jumps[(o, d)] / exposure[o] for (o, d) in jumps.index}   # intensity per origin-state person-year
naive = jumps[("H", "I")] / ins[ins.origin == "H"].exposure_py.sum()
print(f"Pooled exposure (H, I) = {exposure['H']:,}, {exposure['I']:,} person-years; initial lives (H, I) = {initial['H']:,}, {initial['I']:,}")
print(f"Wrong way (exposure summed over both destination rows) would halve the H->I rate: {naive:.4f} vs correct {rate[('H','I')]:.4f}")
print(fmt(pd.DataFrame({"jumps": jumps, "exposure_py": [exposure[o] for o, d in jumps.index], "intensity per person-year": pd.Series(rate)})))

S = ["H", "I", "D"]
Q = np.zeros((3, 3))
for (o, d), v in rate.items(): Q[S.index(o), S.index(d)] = v
for i in range(2): Q[i, i] = -Q[i].sum()
Qdf = pd.DataFrame(Q, index=S, columns=S)
print("Row sums (should be 0):", Q.sum(axis=1).round(12))
print("Off-diagonals non-negative:", bool((Q - np.diag(np.diag(Q)) >= 0).all()), "| diagonals non-positive:", bool((np.diag(Q) <= 0).all()), "| death row all zero:", bool((Q[2] == 0).all()))
fig, ax = plt.subplots(figsize=(8, 5)); ax.axis("off"); ax.set_aspect("equal"); ax.set_xlim(-1.2, 8.2); ax.set_ylim(-4.6, 1.9)
pos = {"H": (0.5, 0), "I": (6.5, 0), "D": (3.5, -3.2)}
circ = {}
for s_, (x, y) in pos.items():
    circ[s_] = plt.Circle((x, y), 0.85, color=NAVY if s_ != "D" else "#555", zorder=3); ax.add_patch(circ[s_])
    ax.text(x, y, {"H": "Healthy", "I": "Ill", "D": "Dead"}[s_], ha="center", va="center", color="white", fontsize=10, zorder=4)
def arrow(a, b, txt, rad, push):
    ax.add_patch(plt.matplotlib.patches.FancyArrowPatch(pos[a], pos[b], connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=18,
                 color=AMBER, lw=2, patchA=circ[a], patchB=circ[b], shrinkA=0, shrinkB=0, zorder=2))
    (x1, y1), (x2, y2) = pos[a], pos[b]; dx, dy = x2 - x1, y2 - y1
    lx, ly = (x1 + x2) / 2 + 0.5 * rad * dy * push[0], (y1 + y2) / 2 - 0.5 * rad * dx * push[0]
    ax.text(lx + push[1], ly + push[2], txt, ha="center", va="center", fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
arrow("H", "I", f"H→I  {rate[('H','I')]:.4f}", 0.28, (1.5, 0, 0)); arrow("I", "H", f"I→H (recovery)  {rate[('I','H')]:.4f}", 0.28, (1.5, 0, 0))
arrow("H", "D", f"H→D  {rate[('H','D')]:.4f}", 0.0, (0, -1.0, 0)); arrow("I", "D", f"I→D  {rate[('I','D')]:.4f}", 0.0, (0, 1.0, 0))
ax.text(3.5, -4.5, "Death is absorbing. Intensities per person-year (hypothetical data)", ha="center", fontsize=9)
print(fmt(Qdf))
save_fig("m07_state_diagram")

# Interpretation for the report (check against your own output): Generator. Per person-year, healthy
# people fall ill at 0.1035 and die at 0.0180, while ill people recover at 0.2452 and die at 0.0673. Every
# row of Q sums to zero, off-diagonals are non-negative and the death row is zero.

# 3(b) Holding times and next-jump probabilities

hold = pd.Series({"Healthy": -1 / Q[0, 0], "Ill": -1 / Q[1, 1]}, name="Mean holding time (years)")
nxt = pd.Series({"Ill -> Healthy (recovery)": Q[1, 0] / -Q[1, 1], "Ill -> Dead": Q[1, 2] / -Q[1, 1]}, name="Next-jump probability from Ill")
P1 = expm(Q)                      # one-year matrix, used only for the explanation below
print(f"P(recovery on the next jump from Ill) = {nxt.iloc[0]:.4f}  versus  P(Healthy after 1 year | Ill now) = {P1[1, 0]:.4f}")
print(fmt(pd.concat([hold, nxt], axis=1)))

# Interpretation for the report (check against your own output): Holding times and next jump. People stay
# healthy for 8.2 years on average and ill for 3.2 years. From the ill state, 78.5% of next jumps are
# recoveries and 21.5% are deaths. That 78.5% only compares the two exits and ignores timing. The one-year
# probability of being healthy (19.8%) is much lower because an ill person may simply stay ill for the
# whole year (74.2%), may die first (6.0%), or may recover and then fall ill again.

Qdf.reset_index(names="from_state").to_csv(OUT / "generator_Q.csv", index=False)


# ====================================================================================================
# SECTION 8: TRANSITION MATRICES, OCCUPANCY AND BENEFIT PAYMENTS  (Questions 3(c) and 3(d))
# ====================================================================================================
# 3(c) Transition matrices and expected occupancy

P1, P2 = expm(Q), expm(2 * Q)                  # Student 8: the matrix exponentials
tol = 1e-10
print("P(1) rows sum to 1:", np.allclose(P1.sum(1), 1, atol=tol), "| P(2):", np.allclose(P2.sum(1), 1, atol=tol))
print("Non-negative:", bool((P1 >= -tol).all() and (P2 >= -tol).all()))
print("Death absorbing (row D = 0,0,1):", np.allclose(P1[2], [0, 0, 1]) and np.allclose(P2[2], [0, 0, 1]))
print(f"P(2) = P(1)^2 within tolerance: {np.allclose(P2, P1 @ P1, atol=tol)} (max abs diff {np.abs(P2 - P1 @ P1).max():.2e})")
n0 = np.array([initial["H"], initial["I"], 0.0])
occ = pd.DataFrame([n0, n0 @ P1, n0 @ P2], index=["Start", "End of year 1", "End of year 2"], columns=S)
occ["Total"] = occ.sum(axis=1)
display_P = pd.concat({"P(1)": pd.DataFrame(P1, index=S, columns=S), "P(2)": pd.DataFrame(P2, index=S, columns=S)})
print(display_P.round(5).to_string())
print(fmt(occ))

# Interpretation for the report (check against your own output): Checks and occupancy. All probabilities
# are non-negative, rows sum to one, death is absorbing, and P(2) equals P(1) squared to a maximum
# difference of 2.2e-16. Starting from 23,656 insured lives (20,080 healthy, 3,576 ill), expected
# occupancy after year 1 is 18,704 healthy, 4,335 ill and 617 dead; after year 2 it is 17,622 healthy,
# 4,783 ill and 1,251 dead.

# 3(d) Benefit payments

BEN = 1000                                                    # USD per covered death, no discounting, survivors stay covered
d1 = occ.loc["End of year 1", "D"] - occ.loc["Start", "D"]
d2 = occ.loc["End of year 2", "D"] - occ.loc["End of year 1", "D"]
claims = pd.DataFrame({"Expected deaths": [d1, d2, d1 + d2], "Expected benefit (USD)": [d1 * BEN, d2 * BEN, (d1 + d2) * BEN]}, index=["Year 1", "Year 2", "Two-year total"])
print(fmt(claims))

# Interpretation for the report (check against your own output): Benefits. Expected benefit payments are
# USD 616,501 in year 1 and USD 634,457 in year 2 (USD 1,250,957 over two years, undiscounted).
# *Limitation of pooling constant intensities:* one set of rates is applied to all districts and all ages
# for both years, so it ignores differences between districts, mortality that rises as the cohort ages,
# and any trend. Year-2 claims are probably understated for an ageing cohort.

occ.reset_index(names="time").to_csv(OUT / "insured_occupancy.csv", index=False)

claims.reset_index(names="period").to_csv(OUT / "expected_claims.csv", index=False)


# ====================================================================================================
# SECTION 9: CLUSTERING DATA PREPARATION AND DIAGNOSTICS  (Questions 4(a) and 4(b))
# ====================================================================================================
# 4(a) Feature table, imputation and standardisation

FEATS = ["share_u15", "share_65p", "nmr_2025", "piped_water_pct"]
feat = prof[FEATS + ["region"]].copy()
print("Missing water:", list(feat.index[feat.piped_water_pct.isna()]))
# Primary rule: kNN (k=3) imputation using the three complete demographic features, so every district is kept.
scaler0 = StandardScaler().fit(feat[FEATS[:3]])
Z3 = scaler0.transform(feat[FEATS[:3]])
imp = KNNImputer(n_neighbors=3).fit_transform(np.column_stack([Z3, feat.piped_water_pct.values]))
X_df = feat[FEATS].copy(); X_df["piped_water_pct"] = imp[:, 3]
alt_region = feat.groupby("region").piped_water_pct.transform("median")
imputed = pd.DataFrame({"kNN imputed (primary)": X_df.loc[feat.piped_water_pct.isna(), "piped_water_pct"],
                        "Region median (alternative)": alt_region[feat.piped_water_pct.isna()],
                        "Overall median (alternative)": feat.piped_water_pct.median()})
print(imputed.to_string())
z = (X_df - X_df.mean()) / X_df.std()
print("\nOutlier check, |z| > 2.5:", [(i, c, round(z.loc[i, c], 2)) for c in FEATS for i in z.index if abs(z.loc[i, c]) > 2.5] or "none")
iqr = X_df.quantile(.75) - X_df.quantile(.25)
print("Outlier check, outside 1.5 x IQR:", [(i, c) for c in FEATS for i in X_df.index if (X_df.loc[i, c] < X_df[c].quantile(.25) - 1.5 * iqr[c]) or (X_df.loc[i, c] > X_df[c].quantile(.75) + 1.5 * iqr[c])] or "none")
scaler = StandardScaler().fit(X_df)
X = scaler.transform(X_df)                                     # district_id is excluded: it is an arbitrary label, not a demographic feature
print("Standardised means ~0:", np.abs(X.mean(0)).max().round(12), "| sds = 1:", X.std(0).round(3))
print(fmt(X_df.describe().T[["mean", "std", "min", "max"]]))

# Interpretation for the report (check against your own output): Data preparation. The two missing water
# values (D04, D17) are filled by kNN imputation (k = 3) on the three complete standardised features, so
# all 24 districts are retained. Water access follows the demographic type in the other 22 districts, so
# this gives D04 35.9% (a young district) and D17 73.6% (a migrant-gaining district), where a region
# median would give both 50.7%. No outliers appear under the |z| > 2.5 or the 1.5 x IQR rule. Features are
# standardised so that net migration per 1,000 and percentages count equally. district_id is excluded
# because it is an arbitrary label: its number carries no demographic meaning and would only add noise to
# the distances.

# 4(b) k-means diagnostics

ks = range(1, 7)
fits = {k: KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(X) for k in ks}
wcss = pd.Series({k: fits[k].inertia_ for k in ks}, name="WCSS")
diag = pd.DataFrame({"WCSS": wcss, "Reduction vs previous k (%)": (-wcss.pct_change() * 100)})
diag["Mean silhouette"] = pd.Series({k: silhouette_score(X, fits[k].labels_) for k in range(2, 7)})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(list(ks), wcss.values, marker="o", color=NAVY); ax[0].set_xlabel("k"); ax[0].set_ylabel("WCSS"); ax[0].set_title("Elbow plot")
ax[1].plot(range(2, 7), diag["Mean silhouette"].dropna().values, marker="o", color=AMBER); ax[1].set_xlabel("k"); ax[1].set_ylabel("Mean silhouette"); ax[1].set_title("Mean silhouette")
fig.suptitle("District clustering diagnostics (hypothetical data)"); plt.tight_layout()
print(fmt(diag))
save_fig("m09_wcss_and_silhouette")

# Interpretation for the report (check against your own output): Reading the diagnostics. WCSS (within-
# cluster sum of squares) always falls as k rises, so we look for the elbow: it drops 55.5% from k = 1 to
# 2, 94.0% from 2 to 3, then only 37.5%, 35.3% and 22.2%. Mean silhouette is 0.595 (k = 2), 0.864 (k = 3),
# 0.783 (k = 4), 0.637 (k = 5) and 0.474 (k = 6). Silhouette values near 1 mean a district sits well
# inside its cluster; values near 0 mean it lies between two clusters; negative values mean it is probably
# in the wrong one.

cands = [3, 4]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for a_, k in zip(axes, cands):
    lab, sil = fits[k].labels_, silhouette_samples(X, fits[k].labels_)
    y = 0
    for c in range(k):
        v = np.sort(sil[lab == c]); a_.barh(range(y, y + len(v)), v, color=plt.cm.tab10(c), height=1.0, label=f"Cluster {c} (n={len(v)})"); y += len(v) + 1
    a_.axvline(sil.mean(), color="red", ls="--"); a_.set_title(f"k = {k}, mean silhouette {sil.mean():.3f}"); a_.set_xlabel("Silhouette"); a_.set_yticks([]); a_.legend(fontsize=7)
fig.suptitle("Individual silhouettes by cluster (hypothetical data)"); plt.tight_layout()
save_fig("m09_silhouettes_by_cluster")

# Interpretation for the report (check against your own output): For both plausible candidates (k = 3 and
# k = 4) every district has a positive silhouette. At k = 4 one natural group is split and the bars
# shorten, which matches the lower mean.

diag.reset_index(names="k").to_csv(OUT / "cluster_diagnostics.csv", index=False)

X_df.reset_index().to_csv(OUT / "clustering_features_used.csv", index=False)


# ====================================================================================================
# SECTION 10: CHOICE OF K, STABILITY, PROFILES AND DASHBOARD EXPORTS  (Question 4(c) and dashboard outputs)
# ====================================================================================================
# 4(c) Choice of k, stability checks and district profiles

K = 3
prim = KMeans(n_clusters=K, n_init=50, random_state=SEED).fit(X)
lab = prim.labels_
# Stability 1: different seed
lab_seed = KMeans(n_clusters=K, n_init=50, random_state=2026).fit(X).labels_
# Stability 2: different missing-value treatment (region median instead of kNN)
Xr = feat[FEATS].copy(); Xr["piped_water_pct"] = Xr.piped_water_pct.fillna(alt_region)
lab_reg = KMeans(n_clusters=K, n_init=50, random_state=SEED).fit(StandardScaler().fit_transform(Xr)).labels_
# Stability 3: different feature selection (drop piped water, so nothing is imputed)
lab_nowater = KMeans(n_clusters=K, n_init=50, random_state=SEED).fit(StandardScaler().fit_transform(X_df[FEATS[:3]])).labels_
stab = pd.DataFrame({"Adjusted Rand index vs primary": [adjusted_rand_score(lab, l) for l in (lab_seed, lab_reg, lab_nowater)],
                     "Districts with a different grouping": [int(len(lab) - sum(pd.crosstab(lab, l).max(axis=1))) for l in (lab_seed, lab_reg, lab_nowater)]},
                    index=["Different seed (2026)", "Region-median imputation", "Drop piped water (3 features)"])
print(fmt(stab))

# Interpretation for the report (check against your own output): Choice of k = 3. k = 3 has the largest
# WCSS reduction (94.0%) and the highest mean silhouette (0.864), and it has a clear demographic meaning
# (three distinct population types, below). The result is stable: the adjusted Rand index against the
# primary fit is 1.0 for a different seed, for region-median imputation and for dropping piped water, so
# no district changes cluster.

res_c = X_df.copy(); res_c["cluster"] = lab; res_c["region"] = feat.region
cent = res_c.groupby("cluster")[FEATS].mean()
name = {}
name[cent.share_u15.idxmax()] = "Young, high-fertility"
name[cent.share_65p.idxmax()] = "Ageing"
name[[c for c in cent.index if c not in name][0]] = "Working-age, in-migration"
res_c["cluster_name"] = res_c.cluster.map(name)
cent["Districts (n)"] = res_c.groupby("cluster").size()
cent["Cluster name"] = cent.index.map(name)
cent.columns = ["Under-15 share", "65+ share", "Net migration per 1,000", "Piped water (%)", "Districts (n)", "Cluster name"]
print("Cluster membership:")
for c, grp in res_c.groupby("cluster"): print(f"  {name[c]}: {', '.join(grp.index)}")
print(fmt(cent))

# Interpretation for the report (check against your own output): Profiles. The three clusters have 8
# districts each. *Young, high-fertility* (D01, D04, D07, D10, D13, D16, D19, D22): 29.0% under 15, net
# out-migration, lowest piped water (36.5%). *Working-age, in-migration* (D02, D05, D08, D11, D14, D17,
# D20, D23): 12.9% under 15, 12.6% aged 65+, net in-migration of 11.9 per 1,000, highest water access
# (78.4%). *Ageing* (D03, D06, D09, D12, D15, D18, D21, D24): 27.9% aged 65+, 9.5% under 15. *Use:* the
# profiles let planners group districts for shared services (schools and maternal health, housing and
# jobs, elder care) instead of 24 separate plans. *Limitation:* with 24 districts, four features and one
# year, the cluster means hide variation within clusters, and the two imputed water values are estimates.

# Dashboard outputs and reconciliation

rt = prof.copy()
x_ = agesex.assign(grp=np.select([agesex.age_start < 15, agesex.age_start >= 65], ["u15", "o65"], "wa")).pivot_table(index="district_id", columns="grp", values="population_2025", aggfunc="sum")
rt["youth_dependency_ratio"] = x_.u15 / x_.wa * 100; rt["old_age_dependency_ratio"] = x_.o65 / x_.wa * 100
rt["piped_water_imputed"] = feat.piped_water_pct.isna(); rt["piped_water_pct_used"] = X_df.piped_water_pct
rt["cluster"] = res_c.cluster; rt["cluster_name"] = res_c.cluster_name
rt.reset_index().to_csv(OUT / "district_profiles.csv", index=False)
res_c.reset_index().to_csv(OUT / "clustering_results.csv", index=False)
cent.reset_index().to_csv(OUT / "cluster_centroids.csv", index=False)
# Reconcile two dashboard figures to the Python results by reading the exported CSVs back
d_tot = pd.read_csv(OUT / "projection_age_sex.csv").query("scenario == 'Base'").population_2030.sum()
d_clm = pd.read_csv(OUT / "expected_claims.csv").set_index("period").loc["Year 1", "Expected benefit (USD)"]
print(f"Base 2030 total: CSV {d_tot:,.1f} vs Python {proj['Base'].values.sum():,.1f} -> match {np.isclose(d_tot, proj['Base'].values.sum())}")
print(f"Year-1 expected claims: CSV {d_clm:,.1f} vs Python {d1 * BEN:,.1f} -> match {np.isclose(d_clm, d1 * BEN)}")
print("Files in outputs/:", sorted(p.name for p in OUT.glob("*.csv")))

print("\nAll sections finished. Results are in outputs/ and figures/.")
