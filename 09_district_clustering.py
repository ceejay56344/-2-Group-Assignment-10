"""Cluster naming and stability. Independently executable HASTS206 part.
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
# ---------- life table
L=lt.copy(); L['n']=L.interval_years; L['mx']=L.deaths/L.person_years
L['qx']=np.where(L.n.notna(), L.n*L.mx/(1+(L.n-L.ax)*L.mx), 1.0)
l=[100000.0]
for i in range(len(L)-1): l.append(l[-1]*(1-L.qx.iloc[i]))
L['lx']=l; L['dx']=L.lx*L.qx
L['Lx']=np.where(L.n.notna(), L.n*(L.lx-L.dx)+L.ax*L.dx, L.lx/L.mx)
L['Tx']=L.Lx[::-1].cumsum()[::-1]; L['ex']=L.Tx/L.lx
L['age_band']=L.age_start.map(lab)
print(L[['age_start','mx','qx','lx','dx','Lx','Tx','ex']].round(4).to_string())
print('check: sum dx =',L.dx.sum(),' e0=',L.ex.iloc[0],' e65=',L.ex[L.age_start==65].iloc[0],' T0 check',L.Lx.sum())
L[['age_band','age_start','person_years','deaths','mx','qx','lx','dx','Lx','Tx','ex']].to_csv(O+'life_table.csv',index=False)
# ---------- projection
sexcols=['F','M']
def cc(dd,scn,fert=None):
    s=sc.loc[scn]; fm=s.fertility_multiplier if fert is None else fert
    d=dd.set_index(['age_start','sex']).copy(); out={}
    for sx in sexcols:
        x=d.xs(sx,level='sex'); ages=list(range(0,90,5)); surv=(x.population_2025*x.survival_5yr)
        new=pd.Series(0.0,index=ages)
        for ag in ages:
            if ag==0: continue
            if ag==85: new[85]=surv[80]+surv[85]
            else: new[ag]=surv[ag-5]
        births=5*dd.births_to_women_2025.sum()*fm
        nb=births*(0.49 if sx=='F' else 0.51)*(s.newborn_survival_f if sx=='F' else s.newborn_survival_m)
        new[0]=nb
        mig=5*x.arrivals_2025*s.arrivals_multiplier-5*x.departures_2025*s.departures_multiplier
        out[sx]=(new+mig, new, mig, births, nb)
    return out
rows=[];tot=[]
for did,dd in a.groupby('district_id'):
    for scn in sc.index:
        o=cc(dd,scn)
        for sx in sexcols:
            fin,pre,mig,births,nb=o[sx]
            for ag in range(0,90,5): rows.append([did,scn,ag,lab[ag],sx,fin[ag]])
        fin_all=sum(o[sx][0].sum() for sx in sexcols)
        u15=sum(o[sx][0][[0,5,10]].sum() for sx in sexcols); o65=sum(o[sx][0][[65,70,75,80,85]].sum() for sx in sexcols)
        a04=sum(o[sx][0][0] for sx in sexcols)
        bt=5*dd.births_to_women_2025.sum()*sc.loc[scn].fertility_multiplier
        tot.append([did,scn,fin_all,u15,o65,100*u15/fin_all,100*o65/fin_all,a04,bt,sum(o[sx][4] for sx in sexcols),sum(o[sx][2].sum() for sx in sexcols)])
pa=pd.DataFrame(rows,columns=['district_id','scenario','age_start','age_band','sex','pop_2030']); pa['sex_label']=pa.sex.map({'F':'Female','M':'Male'})
pt=pd.DataFrame(tot,columns=['district_id','scenario','pop_2030','under15','age65plus','share_u15','share_65p','age_0_4','births_5yr_total','surviving_births','net_migration_5yr'])
pa.to_csv(O+'projection_age_sex.csv',index=False)
# growth models for all districts
g2=p[['district_id','population_2020','pop2025']].copy()
g2['arith_2030']=g2.pop2025+(g2.pop2025-g2.population_2020)
g2['geom_2030']=g2.pop2025*(g2.pop2025/g2.population_2020)
g2['r_geom_pa_pct']=100*((g2.pop2025/g2.population_2020)**(1/5)-1)
pt2=pt.merge(g2[['district_id','pop2025','arith_2030','geom_2030']],on='district_id')
pt2.to_csv(O+'projection_totals.csv',index=False)
print(pt2[pt2.district_id.isin(SEL)].round(1).to_string())
# long format totals for comparison chart
cmp=[]
for _,r in pt2.iterrows():
    pass
base=pt[(pt.district_id==PROJ)&(pt.scenario=='Base')].iloc[0]
# sensitivity: base w/ high fertility only
dd=a[a.district_id==PROJ]; o=cc(dd,'Base',fert=sc.loc['High'].fertility_multiplier)
hi04=sum(o[sx][0][0] for sx in sexcols); tothi=sum(o[sx][0].sum() for sx in sexcols)
print('SENS',PROJ,'base 0-4',base.age_0_4,'->',hi04,'diff',hi04-base.age_0_4, (hi04/base.age_0_4-1)*100,'total diff',tothi-base.pop_2030)
sens=pd.DataFrame([[PROJ,'Base',base.age_0_4,base.pop_2030],[PROJ,'Base + High fertility only',hi04,tothi]],columns=['district_id','case','age_0_4_2030','total_2030']); sens.to_csv(O+'projection_sensitivity.csv',index=False)
# accounting check for PROJ base
o=cc(dd,'Base'); surv_tot=sum((dd[dd.sex==sx].population_2025*dd[dd.sex==sx].survival_5yr).sum() for sx in sexcols)
print('ACC check: survivors',surv_tot,'+surv births',sum(o[sx][4] for sx in sexcols),'+net mig',sum(o[sx][2].sum() for sx in sexcols),'=',surv_tot+sum(o[sx][4] for sx in sexcols)+sum(o[sx][2].sum() for sx in sexcols),' vs total',base.pop_2030)
# ---------- Markov
ex=tr.drop_duplicates(['district_id','origin'])
EH=ex[ex.origin=='H'].exposure_py.sum(); EI=ex[ex.origin=='I'].exposure_py.sum()
cnt=tr.groupby(['origin','destination']).transition_count.sum()
lHI=cnt['H','I']/EH; lHD=cnt['H','D']/EH; lIH=cnt['I','H']/EI; lID=cnt['I','D']/EI
Q=np.array([[-(lHI+lHD),lHI,lHD],[lIH,-(lIH+lID),lID],[0,0,0]])
print('\nQ',Q,'\nrowsums',Q.sum(1),'EH,EI',EH,EI, dict(cnt))
print('hold H',1/(lHI+lHD),'hold I',1/(lIH+lID),'P(next=H|I)',lIH/(lIH+lID),'P(next=D|I)',lID/(lIH+lID),'P(next I|H)',lHI/(lHI+lHD))
P1=expm(Q); P2=expm(2*Q); print('P1',P1.round(4),'\nP2',P2.round(4),'\nmax|P2-P1^2|',abs(P2-P1@P1).max(),'min',P1.min(),'rowsum',P1.sum(1))
init=ex.groupby('origin').initial_lives.sum(); n0=np.array([init['H'],init['I'],0.0]); print('init',n0)
o1=n0@P1; o2=n0@P2; print('occ1',o1,'occ2',o2)
d1=o1[2]-n0[2]; d2=o2[2]-o1[2]; print('deaths y1',d1,'y2',d2,'benefit',1000*d1,1000*d2)
qdf=pd.DataFrame(Q,index=['H','I','D'],columns=['H','I','D']).reset_index().rename(columns={'index':'From'}); qdf.to_csv(O+'markov_generator_Q.csv',index=False)
occ=pd.DataFrame({'time_years':[0,0,0,1,1,1,2,2,2],'state':['Healthy','Ill','Dead']*3,'expected_lives':list(n0)+list(o1)+list(o2)})
occ.to_csv(O+'markov_occupancy.csv',index=False)
cl=pd.DataFrame({'year':['Year 1','Year 2'],'expected_deaths':[d1,d2],'benefit_usd':[1000*d1,1000*d2]}); cl['cumulative_benefit_usd']=cl.benefit_usd.cumsum(); cl.to_csv(O+'markov_claims.csv',index=False)
pm=[]
for t,P in [(1,P1),(2,P2)]:
    for i,fr in enumerate('HID'):
        for j,to in enumerate('HID'): pm.append([t,fr,to,P[i,j]])
pd.DataFrame(pm,columns=['t_years','from_state','to_state','probability']).to_csv(O+'markov_P.csv',index=False)
# ---------- clustering
f=p[['district_id','share_u15','share_65p','NMR','piped_water_pct','region']].copy()
f['water_imputed']=f.piped_water_pct.fillna(f.groupby('region').piped_water_pct.transform('median'))
print(f[f.piped_water_pct.isna()])
feat=['share_u15','share_65p','NMR','water_imputed']
X=(f[feat]-f[feat].mean())/f[feat].std(ddof=0)
res=[]; fits={}
for k in range(1,7):
    km=KMeans(n_clusters=k,n_init=50,random_state=42).fit(X); fits[k]=km
    res.append([k,km.inertia_, silhouette_score(X,km.labels_) if k>1 else np.nan])
cr=pd.DataFrame(res,columns=['k','WCSS','mean_silhouette']); cr['pct_reduction_vs_prev']=100*(1-cr.WCSS/cr.WCSS.shift(1))
print(cr.round(3).to_string()); cr.to_csv(O+'cluster_diagnostics.csv',index=False)
K=3; km=fits[K]; f['cluster_raw']=km.labels_
f['silhouette']=silhouette_samples(X,km.labels_)
cent=f.groupby('cluster_raw')[['share_u15','share_65p','NMR','water_imputed']].mean()
print(f.groupby('cluster_raw').district_id.apply(list)); print(cent.round(2))
# name clusters
names={}
for c,r in cent.iterrows():
    if r.share_65p>20: names[c]='Ageing, low-fertility'
    elif r.share_u15>20: names[c]='Young, low water, out-migration'
    else: names[c]='Working-age, in-migration, high water'
f['cluster']=f.cluster_raw.map(names)
# stability: different seed + mean imputation / drop water
km2=KMeans(n_clusters=3,n_init=50,random_state=7).fit(X); print('ARI seed',adjusted_rand_score(km.labels_,km2.labels_))
f['water_mean']=f.piped_water_pct.fillna(f.piped_water_pct.mean()); X3=(f[['share_u15','share_65p','NMR']]-f[['share_u15','share_65p','NMR']].mean())/f[['share_u15','share_65p','NMR']].std(ddof=0)
km3=KMeans(n_clusters=3,n_init=50,random_state=42).fit(X3); print('ARI drop water',adjusted_rand_score(km.labels_,km3.labels_))
Xm=(f[['share_u15','share_65p','NMR','water_mean']]-f[['share_u15','share_65p','NMR','water_mean']].mean())/f[['share_u15','share_65p','NMR','water_mean']].std(ddof=0)
km4=KMeans(n_clusters=3,n_init=50,random_state=42).fit(Xm); print('ARI mean imp',adjusted_rand_score(km.labels_,km4.labels_))
# k=2 silhouettes for plausibility
print('k2 sil',silhouette_score(X,fits[2].labels_),' k4',silhouette_score(X,fits[4].labels_))
f.merge(p[['district_id','pop2025']],on='district_id').to_csv(O+'cluster_membership.csv',index=False)
cs=f.groupby('cluster').agg(districts=('district_id','count'),share_u15=('share_u15','mean'),share_65p=('share_65p','mean'),NMR=('NMR','mean'),piped_water_pct=('water_imputed','mean')).reset_index(); cs.to_csv(O+'cluster_summary.csv',index=False); print(cs.round(2))
