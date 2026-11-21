"""Population pyramid data. Independently executable HASTS206 part.
Keep this file under assignment_10_parts/ beside data/raw/ in the project root.
"""
from pathlib import Path
import os, sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)
import pandas as pd, numpy as np
from pathlib import Path
from scipy.linalg import expm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
# Locate the CSVs relative to this script, so the program works on a Mac,
# Windows or Linux without the original author's /home/claude path.
BASE = Path(__file__).resolve().parent.parent
DATA_OPTIONS = [BASE / 'data' / 'raw', BASE / 'data']
D = next((folder for folder in DATA_OPTIONS if (folder / 'districts.csv').is_file()), None)
if D is None:
    raise FileNotFoundError(
        'Could not find districts.csv. Extract the complete assignment ZIP and '
        'put this script in the main assignment folder beside data/raw/. '
        f'Checked: {DATA_OPTIONS}'
    )
O = str(BASE / 'outputs') + '/'
Path(O).mkdir(parents=True, exist_ok=True)
dist=pd.read_csv(D / 'districts.csv'); a=pd.read_csv(D / 'age_sex.csv'); sc=pd.read_csv(D / 'projection_scenarios.csv').set_index('scenario')
lt=pd.read_csv(D / 'life_table_inputs.csv'); tr=pd.read_csv(D / 'insured_transitions.csv')
# ---------- DQ
print(len(dist),len(a),a.groupby('district_id').size().unique(),a.duplicated(['district_id','age_start','sex']).sum(), dist.piped_water_pct.isna().sum())
# ---------- profile
g=a.groupby('district_id')
p=pd.DataFrame({'pop2025':g.population_2025.sum(),'u15':a[a.age_start<15].groupby('district_id').population_2025.sum(),
 'o65':a[a.age_start>=65].groupby('district_id').population_2025.sum(),
 'f':a[a.sex=='F'].groupby('district_id').population_2025.sum(),'m':a[a.sex=='M'].groupby('district_id').population_2025.sum(),
 'w1564':a[(a.age_start>=15)&(a.age_start<65)].groupby('district_id').population_2025.sum(),
 'baseline_deaths':g.baseline_deaths_2025.sum(),'arr_agesex':g.arrivals_2025.sum(),'dep_agesex':g.departures_2025.sum(),
 'births_agesex':g.births_to_women_2025.sum()}).reset_index()
p=p.merge(dist,on='district_id')
p['share_u15']=100*p.u15/p.pop2025; p['share_65p']=100*p.o65/p.pop2025
p['sex_ratio_M_per_100F']=100*p.m/p.f; p['youth_dep']=100*p.u15/p.w1564; p['old_dep']=100*p.o65/p.w1564
p['mean_pop']=(p.pop2025+p.population_2026)/2
p['CBR']=1000*p.births_2025/p.mean_pop; p['CDR']=1000*p.deaths_2025/p.mean_pop
p['NMR']=1000*(p.arrivals_2025-p.departures_2025)/p.mean_pop
p['recon_2026']=p.pop2025+p.births_2025-p.deaths_2025+p.arrivals_2025-p.departures_2025
p['recon_diff']=p.recon_2026-p.population_2026
p['deaths_gap_newborn']=p.deaths_2025-p.baseline_deaths
print(p[['district_id','pop2025','population_2026','recon_2026','recon_diff','births_2025','births_agesex','arrivals_2025','arr_agesex','deaths_gap_newborn']].head(6).to_string())
print('births match',(p.births_2025==p.births_agesex).all(),'arr match',(p.arrivals_2025==p.arr_agesex).all(),'dep match',(p.departures_2025==p.dep_agesex).all(), 'recon max abs diff',p.recon_diff.abs().max())
print(p[['district_id','share_u15','share_65p','NMR','CBR','CDR','piped_water_pct','pop2025','population_2020']].round(1).to_string())
p.to_csv(O+'district_profiles.csv',index=False)

print('\n=== CHOICE'); SEL=['D01','D24']; PROJ='D01'
# ---------- pyramid
pyr=a[a.district_id.isin(SEL)].copy()
lab={s:(f'{s}-{s+4}' if s<85 else '85+') for s in range(0,90,5)}
pyr['age_band']=pyr.age_start.map(lab); pyr['population_signed']=np.where(pyr.sex=='M',-pyr.population_2025,pyr.population_2025)
pyr['sex_label']=pyr.sex.map({'F':'Female','M':'Male'})
pyr[['district_id','age_start','age_band','sex_label','population_2025','population_signed']].to_csv(O+'pyramid_selected.csv',index=False)
print(p[p.district_id.isin(SEL)][['district_id','pop2025','sex_ratio_M_per_100F','youth_dep','old_dep','CBR','CDR','NMR','population_2020','population_2026','recon_2026']].round(2).to_string())
