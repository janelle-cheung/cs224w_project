import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
csv_path = os.path.join('data','osfstorage-archive','Details information for healthy subjects.csv')
out_path = os.path.join('plots','stage_composition_per_subject.png')

# Read CSV robustly
df = pd.read_csv(csv_path, skipinitialspace=True)
# Normalize column names
cols = [c.strip() for c in df.columns]
df.columns = cols

# Find columns
def find_col(containing):
    for c in df.columns:
        if containing.lower() in c.lower():
            return c
    raise KeyError(f"No column containing '{containing}'")

subj_col = find_col('Subjects')
TST_col = find_col('TST')
N1_col = find_col('N1')
N2_col = find_col('N2')
N3_col = find_col('N3')
REM_col = None
# REM may be labeled 'R(min)' or 'REML (min)'
for candidate in ['REM','R(','REML']:
    try:
        REM_col = find_col(candidate)
        break
    except KeyError:
        REM_col = None

if REM_col is None:
    raise KeyError('REM column not found')

# Ensure numeric
for c in [TST_col, N1_col, N2_col, N3_col, REM_col]:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Sort by SubjectID to preserve order
subjects = df[subj_col].astype(str).tolist()
N1 = df[N1_col].fillna(0).astype(float)
N2 = df[N2_col].fillna(0).astype(float)
N3 = df[N3_col].fillna(0).astype(float)
REM = df[REM_col].fillna(0).astype(float)
TST = df[TST_col].fillna(1).astype(float)

# Compute percentages (using TST)
N1_pct = 100 * N1 / TST
N2_pct = 100 * N2 / TST
N3_pct = 100 * N3 / TST
REM_pct = 100 * REM / TST

# Plot stacked bar chart (percentages)
plt.figure(figsize=(16,6))
ind = range(len(subjects))

p1 = plt.bar(ind, N1_pct, color='#FDB462')
p2 = plt.bar(ind, N2_pct, bottom=N1_pct, color='#80B1D3')
# bottom for N3 is N1+N2
bottom_n3 = N1_pct + N2_pct
p3 = plt.bar(ind, N3_pct, bottom=bottom_n3, color='#B3DE69')
bottom_rem = bottom_n3 + N3_pct
p4 = plt.bar(ind, REM_pct, bottom=bottom_rem, color='#FB8072')

plt.xticks(ind, subjects, rotation=90, fontsize=8)
plt.ylabel('Percentage of TST (%)')
plt.ylim(0,100)
plt.title('Sleep stage composition per subject (N1, N2, N3, REM)')
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('N1','N2','N3','REM'))
plt.tight_layout()

# Ensure plots directory exists
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300)
print('Saved plot to', out_path)
