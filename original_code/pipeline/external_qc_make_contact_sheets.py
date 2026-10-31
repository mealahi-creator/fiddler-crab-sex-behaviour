# =====================================================================
# BLIND QC (external v2) — PART A: make the contact sheets + QC form
# Run AFTER the final preprocessing run (clean Processed_dataset), BEFORE any prediction.
#
# Reads : MyDrive/Crab_External_v2/Processed_dataset/<sex>/<activity>/<clip>/*.jpg
# Writes: MyDrive/Crab_External_v2/QC/External_v2_QC_contact_sheets.pdf   -> give to the QC scorer
#         MyDrive/Crab_External_v2/QC/External_v2_QC_form.xlsx           -> give to the QC scorer
#         MyDrive/Crab_Final_October/PRIVATE_v2_qc_key_DO_NOT_SHARE.csv  -> keep; never give to the scorer
# Clips are shuffled (fixed seed) and shown only as QC_01, QC_02 ... with no sex, activity or file name.
# =====================================================================
import os, glob, random, numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation
from google.colab import drive
drive.mount('/content/drive')

DATA = '/content/drive/MyDrive/Crab_External_v2/Processed_dataset/'
QC   = '/content/drive/MyDrive/Crab_External_v2/QC/'
KEY  = '/content/drive/MyDrive/Crab_Final_October/PRIVATE_v2_qc_key_DO_NOT_SHARE.csv'
SEED = 20261008
os.makedirs(QC, exist_ok=True)
assert not os.path.exists(KEY), 'QC sheets were already made. Do not regenerate (it would reshuffle the IDs).'

clips = []
for d in sorted(glob.glob(DATA + '*/*/*/')):
    sex, act, name = d.rstrip('/').split('/')[-3:]
    fs = sorted(f for f in os.listdir(d) if f.endswith(('.jpg', '.png')))
    if sex in ('male', 'female') and act in ('waving', 'feeding') and len(fs) >= 2:   # < 2 frames: excluded by plan
        clips.append(dict(seq_name=name, sex=sex, activity=act, path=d, files=fs))
random.Random(SEED).shuffle(clips)
for i, c in enumerate(clips, 1): c['qc_id'] = f'QC_{i:02d}'
print(len(clips), 'clips go to QC')

# ---------- contact sheets: 2 clips per page, 8 evenly spaced frames each ----------
T, GAP, HEAD = 190, 6, 46
try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 28)
except Exception: font = ImageFont.load_default()
def strip(c):
    idx = np.linspace(0, len(c['files']) - 1, 8).astype(int)
    im = Image.new('RGB', (8 * T + 9 * GAP, T + HEAD + GAP), 'white')
    ImageDraw.Draw(im).text((GAP, 8), f"{c['qc_id']}   (frames 1-8 of the clip, left to right)", fill='black', font=font)
    for k, j in enumerate(idx):
        fr = Image.open(os.path.join(c['path'], c['files'][j])).convert('RGB').resize((T, T))
        im.paste(fr, (GAP + k * (T + GAP), HEAD))
    return im
pages = []
for p in range(0, len(clips), 2):
    s = [strip(c) for c in clips[p:p + 2]]
    page = Image.new('RGB', (s[0].width, sum(x.height for x in s) + 20), 'white'); y = 0
    for x in s: page.paste(x, (0, y)); y += x.height + 20
    pages.append(page)
pages[0].save(QC + 'External_v2_QC_contact_sheets.pdf', save_all=True, append_images=pages[1:], resolution=100)

# ---------- QC form ----------
wb = Workbook(); ins = wb.active; ins.title = 'Instructions'
lines = ['Blind quality control - external validation clips (v2)',
 'Purpose: decide which clips are unusable BEFORE any model is run on them. Judge image content only.',
 'You must not have seen any model predictions or results for these clips.', '',
 'How to score',
 '1. Open External_v2_QC_contact_sheets.pdf. Each clip (QC_01, QC_02 ...) shows 8 frames spread across the clip.',
 '2. On the "QC form" sheet answer C1-C4 for every clip with Y or N (yellow cells, drop-down lists).',
 '3. Decision fills automatically: any Y -> Exclude. Write a short note for every Y.',
 '4. Score each clip once, in order, and do not revise earlier answers after finishing.',
 '5. Fill your initials and the date on every row.', '',
 'Exclusion criteria (apply exactly as written)',
 'C1  Mixed subject: two or more crabs fill the box in at least 3 of the 8 frames, so it is unclear which crab is tracked.',
 'C2  False detection: the box shows no crab (mud, leaf, shell, debris, shadow, black area) in at least 3 of the 8 frames.',
 'C3  Lighting artefact: an abrupt brightness jump between consecutive frames that changes the whole image.',
 'C4  Subject not visible: the crab is out of the box, fully hidden, or too blurred to see its body in at least 4 of the 8 frames.',
 'Do NOT exclude for: camera angle, crab size, the crab being hard to classify, part of a claw hidden, or ordinary blur/noise.',
 'Do NOT try to judge sex or behaviour - that is not part of QC.']
for i, l in enumerate(lines, 1): ins.cell(i, 1, l).alignment = Alignment(wrap_text=True)
ins.column_dimensions['A'].width = 130; ins['A1'].font = Font(bold=True, size=12)
ws = wb.create_sheet('QC form')
ws.append(['QC ID', 'C1 Mixed subject', 'C2 False detection', 'C3 Lighting artefact', 'C4 Subject not visible',
           'Decision', 'Notes (required for any Y)', 'Scorer initials', 'Date (YYYY-MM-DD)'])
for c in ws[1]: c.font = Font(bold=True); c.fill = PatternFill('solid', fgColor='DDDDDD'); c.alignment = Alignment(wrap_text=True)
yel = PatternFill('solid', fgColor='FFF2CC')
for r, c in enumerate(clips, 2):
    ws.cell(r, 1, c['qc_id'])
    ws.cell(r, 6, f'=IF(COUNTIF(B{r}:E{r},"Y")>0,"Exclude",IF(COUNTIF(B{r}:E{r},"N")=4,"Include","-"))')
    for col in (2, 3, 4, 5, 7, 8, 9): ws.cell(r, col).fill = yel
n = len(clips) + 1
dv = DataValidation(type='list', formula1='"Y,N"', allow_blank=True); ws.add_data_validation(dv); dv.add(f'B2:E{n}')
ws.cell(n + 2, 5, 'Include'); ws.cell(n + 2, 6, f'=COUNTIF(F2:F{n},"Include")')
ws.cell(n + 3, 5, 'Exclude'); ws.cell(n + 3, 6, f'=COUNTIF(F2:F{n},"Exclude")')
ws.cell(n + 4, 5, 'Not yet scored'); ws.cell(n + 4, 6, f'={len(clips)}-F{n+2}-F{n+3}')
for col, w in zip('ABCDEFGHI', [9, 12, 12, 12, 12, 10, 45, 10, 14]): ws.column_dimensions[col].width = w
ws.freeze_panes = 'B2'
wb.save(QC + 'External_v2_QC_form.xlsx')

pd.DataFrame([{k: c[k] for k in ('qc_id', 'seq_name', 'sex', 'activity')} for c in clips]).to_csv(KEY, index=False)
print('Give the scorer ONLY these two files:\n ', QC + 'External_v2_QC_contact_sheets.pdf\n ', QC + 'External_v2_QC_form.xlsx')
print('Keep private:', KEY)
