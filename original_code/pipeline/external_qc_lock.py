# =====================================================================
# BLIND QC (external v2) — PART B: lock the scorer's answers
# Filled form in Drive:  MyDrive/Crab_External_v2/QC/External_v2_QC_form_answered.xlsx
# Writes:                MyDrive/Crab_Final_October/LOCKED_external_v2_exclusions.csv
# =====================================================================
import os, datetime, pandas as pd
from google.colab import drive
drive.mount('/content/drive')

QC     = '/content/drive/MyDrive/Crab_External_v2/QC/'
MODELS = '/content/drive/MyDrive/Crab_Final_October/'
KEY    = MODELS + 'PRIVATE_v2_qc_key_DO_NOT_SHARE.csv'
OUTF   = MODELS + 'LOCKED_external_v2_exclusions.csv'
assert not os.path.exists(OUTF), 'QC already locked - do not overwrite.'

# ---- read the filled form (any sheet name; finds the sheet with a "QC ID" column) ----
xl = pd.ExcelFile(QC + 'External_v2_QC_form_answered.xlsx')
print('Sheets in the answered file:', xl.sheet_names)
sheet = next((s for s in xl.sheet_names
              if 'QC ID' in [str(c).strip() for c in pd.read_excel(xl, sheet_name=s, nrows=0).columns]), None)
assert sheet, 'No sheet with a "QC ID" column - this is not the filled v2 QC form.'
f = pd.read_excel(xl, sheet_name=sheet)
f.columns = [str(c).strip() for c in f.columns]

# ---- accept small changes in column headings ----
def find(start):
    hit = [c for c in f.columns if c.lower().startswith(start)]
    assert hit, f'No column starting with "{start}" - columns are: {list(f.columns)}'
    return hit[0]
f = f.rename(columns={find('c1'): 'C1', find('c2'): 'C2', find('c3'): 'C3', find('c4'): 'C4',
                      find('notes'): 'notes', find('scorer'): 'scorer', find('date'): 'qc_date'})
f = f[f['QC ID'].astype(str).str.strip().str.startswith('QC_')].copy()
f['QC ID'] = f['QC ID'].astype(str).str.strip()
print(f'Using sheet "{sheet}" with {len(f)} QC rows')

# ---- completeness checks ----
crit = ['C1', 'C2', 'C3', 'C4']
for c in crit: f[c] = f[c].astype(str).str.strip().str.upper()
bad = f[~f[crit].isin(['Y', 'N']).all(axis=1)]
assert bad.empty, f'Unscored or invalid cells (must be Y or N) in: {bad["QC ID"].tolist()}'
f['blind_exclude'] = (f[crit] == 'Y').any(axis=1)
nonote = f[f.blind_exclude & f['notes'].isna()]
assert nonote.empty, f'Exclusions without a note: {nonote["QC ID"].tolist()}'
assert f['scorer'].notna().all(), f'Scorer initials missing in: {f[f.scorer.isna()]["QC ID"].tolist()}'
assert f['qc_date'].notna().all(), f'Date missing in: {f[f.qc_date.isna()]["QC ID"].tolist()}'

# ---- match with the private key (must be the NEW v2 clips) ----
key = pd.read_csv(KEY)
assert len(f) == len(key) and set(f['QC ID']) == set(key.qc_id), \
    f'Form and key do not match: form has {len(f)} rows, key has {len(key)} clips. Use the new v2 form and PDF.'
out = key.merge(f.rename(columns={'QC ID': 'qc_id'}), on='qc_id', how='left')
out['qc_date'] = out['qc_date'].astype(str)
out['locked_utc'] = datetime.datetime.utcnow().isoformat(timespec='seconds')
out[['seq_name', 'qc_id', 'C1', 'C2', 'C3', 'C4', 'blind_exclude', 'notes', 'scorer', 'qc_date', 'locked_utc']].to_csv(OUTF, index=False)
print(f'LOCKED: {len(out)} clips, {int(out.blind_exclude.sum())} excluded, {int((~out.blind_exclude).sum())} kept -> {OUTF}')
print('Now run external_test_v2_once_colab.py once.')
