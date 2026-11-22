## Repository contents

```
assignment 2.py                          complete analysis in one script
HASTS206\\\_Assignment2\\\_Analysis.ipynb      editable notebook (same code)
HASTS206\\\_Assignment2\\\_Analysis.html       executed notebook export
README.md                                this file
requirements.txt                         libraries needed
data/raw/                                ORIGINAL supplied files (never edited)
    districts.csv  age\\\_sex.csv  life\\\_table\\\_inputs.csv
    projection\\\_scenarios.csv  insured\\\_transitions.csv
    README.txt                           the lecturer's data guide
outputs/                                 created by the script: CSVs read by Power BI
figures/                                 created by the script: charts (PNG)
```

## Dependencies

* Python 3.10 or newer
* numpy, pandas, scipy, scikit-learn, matplotlib

Tested with Python 3.12.3, numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1 and matplotlib 3.11.2. Other recent versions should work.

`requirements.txt` contains:

```
numpy
pandas
scipy
scikit-learn
matplotlib
```

## How to rerun the analysis

1. **Get the files.** Clone the repository, or download it as a ZIP and extract the whole folder. Check that `data/raw/` holds the five CSV files with exactly these names: `districts.csv`, `age\\\_sex.csv`, `life\\\_table\\\_inputs.csv`, `projection\\\_scenarios.csv`, `insured\\\_transitions.csv`.
2. **Open a terminal in the repository folder.**
3. **(Optional) create a virtual environment.**

   * Windows: `python -m venv .venv` then `.venv\\\\Scripts\\\\activate`
   * Mac or Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. **Install the dependencies:** `pip install -r requirements.txt`
5. **Run the script:** `python "assignment 2.py"`

In PyCharm: open the repository folder, install the libraries from the Terminal tab, then right-click `assignment 2.py` and choose **Run**.

The run takes a few seconds and prints tables. It ends with:

```
All sections finished. Results are in outputs/ and figures/.
```

The script finds `data/raw/` relative to its own location, so it works from any working directory and on Windows, Mac or Linux. Running it again overwrites `outputs/` and `figures/` with identical results.

## How to regenerate the executed HTML

1. Install Jupyter once: `pip install jupyter nbconvert`
2. From the repository folder run:

```
jupyter nbconvert --to html --execute HASTS206\\\_Assignment2\\\_Analysis.ipynb
```

This reruns every cell and writes a fresh `HASTS206\\\_Assignment2\\\_Analysis.html` showing the code, tables and plots.

## What the analysis does

|Section|Content|Brief|
|-|-|-|
|1|Data-quality log and reconciliation|Before analysis|
|2|District selection, population pyramids, sex ratio, dependency ratios|Q1(a)|
|3|Population balance, birth, death and net migration rates|Q1(b)|
|4|Abridged life table|Q1(c)|
|5|Cohort-component projection (Low, Base, High) with accounting checks|Q2(b)|
|6|Arithmetic and geometric growth, comparison, fertility sensitivity|Q2(a), Q2(c)|
|7|Three-state Markov model: generator matrix, diagram, holding times|Q3(a), Q3(b)|
|8|Transition matrices, expected occupancy, benefit payments|Q3(c), Q3(d)|
|9|Clustering preparation and k-means diagnostics|Q4(a), Q4(b)|
|10|Choice of k, stability checks, cluster profiles, dashboard exports|Q4(c)|

Section 5 runs before Section 6 because the Q2(c) comparison needs the cohort-component results.

## CSV outputs consumed by the Power BI dashboard (`outputs/`)

|File|Used for|
|-|-|
|`district\\\_profiles.csv`, `district\\\_selection.csv`, `ratios\\\_selected\\\_districts.csv`, `balance\\\_and\\\_rates.csv`|District profiles page|
|`projection\\\_age\\\_sex.csv`, `projection\\\_totals.csv`, `growth\\\_models.csv`, `fertility\\\_sensitivity.csv`, `projection\\\_accounting\\\_checks.csv`|Projection scenarios page|
|`insured\\\_occupancy.csv`, `expected\\\_claims.csv`, `generator\\\_Q.csv`|Insured occupancy and expected claims page|
|`clustering\\\_results.csv`, `cluster\\\_centroids.csv`, `cluster\\\_diagnostics.csv`, `clustering\\\_features\\\_used.csv`|Clustering page|
|`life\\\_table.csv`, `data\\\_quality\\\_log.csv`|Supporting tables|

To update the dashboard after a rerun, open the `.pbix` file and choose **Refresh**.

**Reconciled dashboard figures** (Power BI value equals the Python value):

* Base 2030 population of D01: 180,173.3
* Year-1 expected claims: USD 616,500.6

## How to check that the rerun worked

|Check|Expected value|
|-|-|
|Balancing equation difference (D01 and D24)|0|
|Life expectancy at birth and at age 65|73.11 and 18.10 years|
|Base 2030 population, D01|180,173.3|
|Fertility High effect on ages 0-4|+2,301.8 (+12.2%)|
|Generator matrix row sums|0|
|Expected benefits, year 1 and year 2|USD 616,501 and USD 634,457|
|Best number of clusters|k = 3 (mean silhouette 0.864)|
|Cluster stability (adjusted Rand index)|1 for all three checks|
|Last two reconciliation lines|both `match True`|

* 

