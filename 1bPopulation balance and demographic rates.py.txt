

@cached
def run():
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

    return {k: v for k, v in locals().items() if not k.startswith("_")}


if __name__ == "__main__":
    run()


Run on its own from the repository root:  python -m analysis.m03_balance_rates
Outputs go to outputs/ and figures/. Upstream results are reused from the modules imported below.
"""
from analysis.common import *


@cached
def run():
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

    return {k: v for k, v in locals().items() if not k.startswith("_")}


if __name__ == "__main__":
    run()
