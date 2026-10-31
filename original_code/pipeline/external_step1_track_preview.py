# =====================================================================
# STEP 1 — Run YOUR YOLO + BoT-SORT on each raw external video and make
#          a preview video with every crab boxed and numbered.
#
# Same detector settings as Video_Preprocessing_Script:
#   weights-yolov11-model-53.pt, grayscale frame resized to 640x640,
#   conf 0.5, botsort.yaml, persist=True, every frame.
#
# Output per video (in OUT):
#   <video>_tracks.csv      every crab box in every frame (full-resolution pixels)
#   <video>_preview.mp4     half-size preview: boxes, crab IDs, time in seconds
# Re-running skips videos that are already done.
# Colab: Runtime > Change runtime type > GPU (much faster).
# =====================================================================
!pip -q install ultralytics lap
import os, re, glob, cv2, numpy as np, pandas as pd, subprocess
from ultralytics import YOLO
from google.colab import drive
drive.mount('/content/drive')

BASE    = '/content/drive/MyDrive/Crab behaviour analysis_UniMAP & NSTRU/'
WEIGHTS = BASE + 'weights-yolov11-model-53.pt'
RAW_DIR = BASE + 'Video/Video_Aj _Tina/22nd Feb 2026/'
OUT     = '/content/drive/MyDrive/Crab_External_v2/tracks/'
ONLY    = []            # e.g. ['MVI_0801', 'MVI_0802'] to do just these; empty = all videos
os.makedirs(OUT, exist_ok=True)

def vid_key(path):
    m = re.match(r'(MVI|IMG)_\d+', os.path.basename(path))
    return m.group(0) if m else os.path.splitext(os.path.basename(path))[0]

videos = sorted(p for p in glob.glob(RAW_DIR + '**/*', recursive=True)
                if p.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')))
if ONLY: videos = [v for v in videos if vid_key(v) in ONLY]
print(len(videos), 'videos to process')

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
          (0, 128, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128)]

for v in videos:
    key = vid_key(v)
    csv_path, prev_path = OUT + f'{key}_tracks.csv', OUT + f'{key}_preview.mp4'
    if os.path.exists(csv_path) and os.path.exists(prev_path):
        print('skip', key); continue
    model = YOLO(WEIGHTS)                      # fresh model = fresh tracker = IDs start again for each video
    cap = cv2.VideoCapture(v); fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = W / 640, H / 640
    tmp = '/content/_preview_tmp.mp4'
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W // 2, H // 2))
    rows, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        g = cv2.cvtColor(cv2.resize(frame, (640, 640)), cv2.COLOR_BGR2GRAY)
        res = model.track(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), conf=0.5, tracker='botsort.yaml',
                          persist=True, verbose=False)
        b = res[0].boxes
        if b is not None and b.id is not None:
            for tid, (x1, y1, x2, y2), c in zip(b.id.cpu().numpy().astype(int), b.xyxy.cpu().numpy(), b.conf.cpu().numpy()):
                X1, Y1, X2, Y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
                rows.append(dict(video=key, frame=i, t=round(i / fps, 3), track_id=int(tid), conf=round(float(c), 3),
                                 x1=X1, y1=Y1, x2=X2, y2=Y2))
                col = COLORS[int(tid) % len(COLORS)]
                cv2.rectangle(frame, (X1, Y1), (X2, Y2), col, 4)
                cv2.putText(frame, f'ID {tid}', (X1, max(40, Y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 4)
        cv2.rectangle(frame, (0, 0), (560, 70), (0, 0, 0), -1)
        cv2.putText(frame, f'{key}  t = {i / fps:6.2f} s', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        writer.write(cv2.resize(frame, (W // 2, H // 2)))
        i += 1
    cap.release(); writer.release()
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', tmp, '-c:v', 'libx264', '-crf', '23', prev_path], check=True)
    ids = pd.DataFrame(rows).track_id.nunique() if rows else 0
    print(f'{key}: {i} frames, {len(rows)} boxes, {ids} crab IDs  ->  {prev_path}')
print('Done. Open the *_preview.mp4 files in VLC and fill External_bout_annotation_v2.xlsx')
