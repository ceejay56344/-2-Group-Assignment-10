"""Student 5: Growth models, comparison with scenarios and fertility sensitivity Questions 2(a) and 2(c). Run on its own from the repository root: python -m

analysis.m05_growth_and_comparison Outputs go to outputs/ and figures/. Upstream results are reused from the modules imported below. """
from analysis.common import *
from analysis import m06_cohort_component


@cached
def run():
    _m06 = m06_cohort_component.run()
    acct = _m06["acct"]
    proj = _m06["proj"]
    project = _m06["project"]
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

    return {k: v for k, v in locals().items() if not k.startswith("_")}


if __name__ == "__main__":
    run()