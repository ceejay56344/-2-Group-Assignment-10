"""Data quality and row checks. Independently executable HASTS206 part.
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
