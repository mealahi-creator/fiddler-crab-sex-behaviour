# =====================================================================
# STEP 0 — Make the new phone videos look like the training videos
#          BEFORE step 1 and step 2.
#
# Why: these phone videos have a VARIABLE frame rate (e.g. "60 fps" but
#   really ~54 fps, "33 fps" but really ~18 fps) and some are 4K portrait.
#   Step 1 counts frames and step 2 cuts by seconds; with a variable frame
#   rate those two clocks disagree, so the cut would miss the bout.
#
# What it does for every video in RAW_IN:
#   - measures the real average frame rate (frames / duration)
#   - if it is below MIN_FPS -> NOT converted, written to the log
#     (too few frames to see a claw wave; decided before any annotation)
#   - otherwise -> constant 30 fps, rotation applied, longer side 1920 px,
#     saved to RAW_OUT with the same name
# Then run step 1 and step 2 (waving versions), which read RAW_OUT.
# =====================================================================
import os, glob, json, subprocess, pandas as pd
from google.colab import drive
drive.mount('/content/drive')

RAW_IN  = '/content/drive/MyDrive/Crab_External_v2/Rawdata/waving/'
RAW_OUT = '/content/drive/MyDrive/Crab_External_v2/Rawdata/waving_30fps/'
FPS, MIN_FPS, LONG_SIDE = 30, 25, 1920
os.makedirs(RAW_OUT, exist_ok=True)

def probe(p):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets',
                          '-show_entries', 'stream=width,height,nb_read_packets:format=duration',
                          '-of', 'json', p], capture_output=True, text=True, check=True).stdout
    j = json.loads(out); s = j['streams'][0]
    return int(s['width']), int(s['height']), int(s['nb_read_packets']), float(j['format']['duration'])

videos = sorted(p for p in glob.glob(RAW_IN + '*') if p.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')))
assert videos, 'No videos found in ' + RAW_IN
log = []
for v in videos:
    name = os.path.basename(v)
    w, h, n, dur = probe(v)
    real_fps = n / dur
    rec = dict(video=name, width=w, height=h, frames=n, duration_s=round(dur, 2), real_fps=round(real_fps, 1))
    if real_fps < MIN_FPS:
        rec.update(status='EXCLUDED', reason=f'real frame rate {real_fps:.1f} < {MIN_FPS} fps')
        print('EXCLUDE', name, rec['reason']); log.append(rec); continue
    out = os.path.join(RAW_OUT, os.path.splitext(name)[0] + '.mp4')
    if not os.path.exists(out):
        # ffmpeg applies the phone's rotation flag automatically; scale so the longer side is 1920 px
        scale = f"scale='if(gt(iw,ih),{LONG_SIDE},-2)':'if(gt(iw,ih),-2,{LONG_SIDE})'"
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', v, '-vf', f'{scale},fps={FPS}',
                        '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'slow', out], check=True)
    w2, h2, n2, d2 = probe(out)
    rec.update(status='OK', out_width=w2, out_height=h2, out_frames=n2, out_fps=round(n2 / d2, 1))
    print(f'OK  {name}: {w}x{h} ~{real_fps:.0f} fps  ->  {w2}x{h2} {n2 / d2:.0f} fps')
    log.append(rec)

pd.DataFrame(log).to_csv(RAW_OUT + 'normalize_log.csv', index=False)
print('\nLog saved:', RAW_OUT + 'normalize_log.csv')
print('Next: step 1 and step 2 (waving versions) with RAW_DIR =', RAW_OUT)
