import os, cv2, uuid, random, mimetypes, smtplib, threading, time
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.status import HTTP_401_UNAUTHORIZED

# ================= CONFIG =================
IMG_SIZE = 172
NUM_FRAMES = 8
FRAME_STEP = 4

class_names = ["not_suspicious", "suspicious"]
id_to_name = {i: name for i, name in enumerate(class_names)}

MODEL_PATH = "movinet_violence_best.keras"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----- EMAIL CONFIG -----
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
ALERT_EMAIL_TO = "kushalsj7@gmail.com"
ALERT_THRESHOLD = 0.80
MAX_ATTACHMENT_MB = 22
DAILY_SUMMARY_HOUR = 21

# ----- ADMIN -----
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"
security = HTTPBasic()

# ----- STORAGE -----
dashboard_logs: List[dict] = []
daily_alert_buffer: List[dict] = []

app = FastAPI(title="Suspicious Activity Detection")

# ================= LOAD MODELS =================
print("Loading classifier...")
clf_model = keras.models.load_model(MODEL_PATH)

print("Loading MoViNet...")
hub_url = "https://tfhub.dev/tensorflow/movinet/a0/base/kinetics-600/classification/3"
movinet_sig = hub.load(hub_url).signatures["serving_default"]

# ================= AUTH =================
def get_current_admin(credentials: HTTPBasicCredentials = Depends(security)):
    if not (credentials.username == ADMIN_USERNAME and credentials.password == ADMIN_PASSWORD):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# ================= VIDEO PREP =================
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    need = 1 + (NUM_FRAMES - 1) * FRAME_STEP
    start = 0 if need > total else random.randint(0, total - need)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    for _ in range(NUM_FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame.astype("float32") / 255.0
        frames.append(frame)
        for _ in range(FRAME_STEP - 1):
            cap.read()
    cap.release()

    while len(frames) < NUM_FRAMES:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))

    return np.expand_dims(np.array(frames), axis=0)

def predict_video(video_path):
    clip = extract_frames(video_path)
    outputs = movinet_sig(image=tf.convert_to_tensor(clip))
    features = outputs["classifier_head"].numpy()
    probs = clf_model.predict(features)[0]
    pred_id = int(np.argmax(probs))
    return id_to_name[pred_id], float(probs[pred_id])

# ================= EMAIL =================
def _smtp_send(msg):
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
        s.starttls()
        s.login(EMAIL_FROM, EMAIL_PASSWORD)
        s.send_message(msg)

def send_email_alert(video_path, filename, pred, conf):
    msg = EmailMessage()
    msg["Subject"] = "🚨 Suspicious Activity Detected"
    msg["From"] = EMAIL_FROM
    msg["To"] = ALERT_EMAIL_TO

    body = f"""
ALERT:

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
File: {filename}
Prediction: {pred}
Confidence: {conf:.2f}
"""
    msg.set_content(body)

    size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if size_mb <= MAX_ATTACHMENT_MB:
        ctype = mimetypes.guess_type(video_path)[0] or "application/octet-stream"
        m, s = ctype.split("/", 1)
        with open(video_path, "rb") as f:
            msg.add_attachment(f.read(), maintype=m, subtype=s, filename=filename)
    _smtp_send(msg)

def send_daily_summary():
    if not daily_alert_buffer:
        return
    msg = EmailMessage()
    msg["Subject"] = "📊 Daily Suspicious Activity Summary"
    msg["From"] = EMAIL_FROM
    msg["To"] = ALERT_EMAIL_TO
    lines = ["DAILY ALERT SUMMARY\n"]
    for r in daily_alert_buffer:
        lines.append(f"{r['time']} | {r['filename']} | {r['confidence']}")
    msg.set_content("\n".join(lines))
    _smtp_send(msg)
    daily_alert_buffer.clear()

def _seconds_until(hour):
    now = datetime.now()
    nxt = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if nxt <= now:
        nxt += timedelta(days=1)
    return (nxt - now).total_seconds()

def summary_worker():
    while True:
        time.sleep(_seconds_until(DAILY_SUMMARY_HOUR))
        send_daily_summary()

threading.Thread(target=summary_worker, daemon=True).start()

# ================= PAGES =================

@app.get("/", response_class=HTMLResponse)
def home():
    recent = dashboard_logs[:10]
    cards = ""
    for r in recent:
        clr = "#ff4d4d" if r["prediction"] == "suspicious" else "#2ecc71"
        cards += (
            f"<div class='card' style='border-left:5px solid {clr};'>"
            f"<div class='card-title'>{r['filename']}</div>"
            f"<div class='card-body'><span class='pill' style='background:{clr}22;color:{clr};'>"
            f"{r['prediction']} ({r['confidence']})</span></div>"
            f"<div class='card-footer'>{r['time']} • Alert: {r['alert']}</div>"
            f"</div>"
        )

    return f"""
<html>
<head>
    <title>Suspicious Activity Detection</title>
    <style>
        body {{
            margin: 0;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: radial-gradient(circle at top, #1f2a3a 0, #050811 55%, #020308 100%);
            color: #f5f7fb;
        }}
        .navbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 40px;
            background: rgba(5, 10, 25, 0.95);
            box-shadow: 0 8px 20px rgba(0,0,0,0.5);
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        .brand {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .logo-mark {{
            width: 38px;
            height: 38px;
            border-radius: 12px;
            background: linear-gradient(135deg,#00e6e6,#007bff);
            display:flex;
            align-items:center;
            justify-content:center;
            font-size:20px;
        }}
        .brand-title {{
            font-size: 22px;
            font-weight: 700;
            letter-spacing: 0.03em;
        }}
        .nav-links a {{
            margin-left: 18px;
            text-decoration: none;
            color: #f5f7fb;
            font-size: 14px;
            padding: 8px 14px;
            border-radius: 999px;
            border: 1px solid transparent;
        }}
        .nav-links a.primary {{
            border-color: #00e6e6;
            background: rgba(0,230,230,0.1);
        }}
        .nav-links a:hover {{
            background: rgba(255,255,255,0.06);
        }}
        .container {{
            max-width: 1100px;
            margin: 40px auto 60px;
            padding: 0 20px;
        }}
        .hero-title {{
            font-size: 34px;
            font-weight: 800;
            margin-bottom: 6px;
        }}
        .hero-subtitle {{
            font-size: 15px;
            color: #c0c4d6;
            margin-bottom: 20px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 18px;
            margin-top: 18px;
        }}
        .card {{
            background: linear-gradient(145deg,#101826,#0a0f18);
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.7);
        }}
        .card-title {{
            font-weight: 600;
            font-size: 15px;
            margin-bottom: 6px;
        }}
        .card-body {{
            margin: 4px 0 10px;
            font-size: 13px;
        }}
        .pill {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .card-footer {{
            font-size: 11px;
            color: #9aa0b5;
        }}
        .actions {{
            margin-top: 24px;
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }}
        a.button {{
            display:inline-flex;
            align-items:center;
            justify-content:center;
            padding:11px 22px;
            background: linear-gradient(135deg,#00e6e6,#007bff);
            color:#050811;
            border-radius:999px;
            text-decoration:none;
            font-weight:600;
            font-size:14px;
            box-shadow:0 12px 30px rgba(0,0,0,0.6);
        }}
        a.button.secondary {{
            background: transparent;
            color:#00e6e6;
            border:1px solid #00e6e6;
            box-shadow:none;
        }}
    </style>
</head>
<body>
    <div class="navbar">
        <div class="brand">
            <div class="logo-mark">👁️</div>
            <div class="brand-title">Suspicious Activity Detection</div>
        </div>
        <div class="nav-links">
            <a href="/predict_page" class="primary">Upload &amp; Predict</a>
            <a href="/dashboard">Admin Dashboard</a>
        </div>
    </div>

    <div class="container">
        <div class="hero">
            <div class="hero-title">Recent Detection Activity</div>
            <div class="hero-subtitle">
                Live overview of the last 10 analysed videos with prediction and alert status.
            </div>
            <div class="actions">
                <a class="button" href="/predict_page">➤ Start New Prediction</a>
                <a class="button secondary" href="/dashboard">🔒 Open Admin Dashboard</a>
            </div>
        </div>

        <div class="grid">
            {cards if cards else "<div class='card'><div class='card-title'>No records yet</div><div class='card-body'>Upload a video to see detections here.</div></div>"}
        </div>
    </div>
</body>
</html>
"""

# ---------- Upload / prediction page ----------
@app.get("/predict_page", response_class=HTMLResponse)
def predict_page():
    return """
<html>
<head>
    <title>Upload Video - Suspicious Activity Detection</title>
    <style>
        body {
            margin:0;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: radial-gradient(circle at top, #1f2a3a 0, #050811 55%, #020308 100%);
            color:#f5f7fb;
            min-height:100vh;
            display:flex;
            align-items:center;
            justify-content:center;
        }
        .card {
            width:480px;
            background:linear-gradient(145deg,#101826,#050a12);
            padding:26px 30px 24px;
            border-radius:18px;
            box-shadow:0 18px 40px rgba(0,0,0,0.8);
            text-align:left;
            border:1px solid rgba(255,255,255,0.06);
        }
        .title {
            font-size:26px;
            font-weight:800;
            margin-bottom:6px;
        }
        .subtitle {
            font-size:13px;
            color:#c0c4d6;
            margin-bottom:20px;
        }
        .field-label {
            font-size:13px;
            margin-bottom:6px;
        }
        input[type="file"] {
            width:100%;
            margin-bottom:18px;
            font-size:13px;
        }
        .buttons {
            display:flex;
            gap:10px;
            flex-wrap:wrap;
            margin-top:4px;
        }
        button {
            padding:10px 20px;
            border-radius:999px;
            border:none;
            background:linear-gradient(135deg,#00e6e6,#007bff);
            color:#050811;
            font-weight:600;
            font-size:14px;
            cursor:pointer;
            box-shadow:0 10px 28px rgba(0,0,0,0.7);
        }
        a.link {
            text-decoration:none;
            font-size:13px;
            color:#00e6e6;
            margin-top:10px;
            display:inline-block;
        }
    </style>
</head>
<body>
    <div class="card">
        <div class="title">Upload Video</div>
        <div class="subtitle">
            Select a short CCTV clip. The system will analyse it for suspicious activity
            and trigger an alert if necessary.
        </div>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <div class="field-label">Video file</div>
            <input type="file" name="file" required>
            <div class="buttons">
                <button type="submit">Analyse Video</button>
            </div>
        </form>
        <a href="/" class="link">← Back to Home</a>
    </div>
</body>
</html>
"""

# ---------- Prediction result ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1]
    temp = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.{ext}")
    with open(temp, "wb") as f:
        f.write(await file.read())

    pred, conf = predict_video(temp)
    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": file.filename,
        "prediction": pred,
        "confidence": round(conf, 3),
        "alert": "NO",
    }

    if pred == "suspicious" and conf >= ALERT_THRESHOLD:
        send_email_alert(temp, file.filename, pred, conf)
        record["alert"] = "SENT"
        daily_alert_buffer.append(record.copy())

    dashboard_logs.insert(0, record)
    os.remove(temp)

    color = "#ff4d4d" if pred == "suspicious" else "#2ecc71"
    nice_pred = pred.replace("_", " ").title()

    html = f"""
<html>
<head>
    <title>Prediction Result - Suspicious Activity Detection</title>
    <style>
        body {{
            margin: 0;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: radial-gradient(circle at top, #1f2a3a 0, #050811 55%, #020308 100%);
            color: #f5f7fb;
            display:flex;
            align-items:center;
            justify-content:center;
            min-height:100vh;
        }}
        .card {{
            width: 540px;
            background: linear-gradient(145deg,#101826,#050a12);
            padding: 28px 30px 26px;
            border-radius: 18px;
            box-shadow: 0 18px 40px rgba(0,0,0,0.8);
            text-align:left;
            border:1px solid rgba(255,255,255,0.06);
        }}
        .title-row {{
            display:flex;
            align-items:center;
            justify-content:space-between;
            margin-bottom:10px;
        }}
        .title {{
            font-size: 26px;
            font-weight: 800;
        }}
        .badge {{
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            background: rgba(255,255,255,0.06);
        }}
        .prediction-pill {{
            display:inline-flex;
            align-items:center;
            gap:8px;
            padding:10px 16px;
            border-radius:14px;
            font-size:15px;
            font-weight:600;
            background: {color}22;
            color:{color};
            margin:12px 0;
        }}
        .label {{
            font-size:13px;
            text-transform:uppercase;
            letter-spacing:0.08em;
            color:#9aa0b5;
        }}
        .value-main {{
            font-size:22px;
            font-weight:700;
        }}
        .meta {{
            font-size:13px;
            color:#c0c4d6;
            margin-top:6px;
        }}
        .buttons {{
            margin-top:22px;
            display:flex;
            gap:12px;
            flex-wrap:wrap;
        }}
        a.btn {{
            text-decoration:none;
            padding:10px 20px;
            border-radius:999px;
            font-size:14px;
            font-weight:600;
            display:inline-flex;
            align-items:center;
            justify-content:center;
            border:1px solid transparent;
        }}
        a.btn-primary {{
            background: linear-gradient(135deg,#00e6e6,#007bff);
            color:#050811;
            box-shadow:0 10px 28px rgba(0,0,0,0.7);
        }}
        a.btn-secondary {{
            background:transparent;
            border-color:#00e6e6;
            color:#00e6e6;
        }}
    </style>
</head>
<body>
    <div class="card">
        <div class="title-row">
            <div class="title">Prediction Result</div>
            <div class="badge">Suspicious Activity Detection</div>
        </div>
        <div class="label">File analysed</div>
        <div class="value-main">{file.filename}</div>

        <div class="prediction-pill">
            <span>Prediction:</span>
            <strong>{nice_pred}</strong>
        </div>

        <div class="label">Confidence</div>
        <div class="value-main">{conf:.2f}</div>
        <div class="meta">
            Alert status: <b>{record['alert']}</b>
        </div>

        <div class="buttons">
            <a href="/predict_page" class="btn btn-primary">Analyse Another Video</a>
            <a href="/" class="btn btn-secondary">Back to Home</a>
            <a href="/dashboard" class="btn btn-secondary">Open Admin Dashboard</a>
        </div>
    </div>
</body>
</html>
"""
    return HTMLResponse(html)

# ---------- Admin dashboard ----------
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(admin: str = Depends(get_current_admin)):
    rows = ""
    for r in dashboard_logs:
        clr = "#ff4d4d" if r["prediction"] == "suspicious" else "#2ecc71"
        rows += (
            f"<tr>"
            f"<td>{r['time']}</td>"
            f"<td>{r['filename']}</td>"
            f"<td style='color:{clr};font-weight:600;'>{r['prediction']}</td>"
            f"<td>{r['confidence']}</td>"
            f"<td>{r['alert']}</td>"
            f"</tr>"
        )

    return f"""
<html>
<head>
    <title>Suspicious Activity Detection - Admin Dashboard</title>
    <style>
        body {{
            margin:0;
            background:#050811;
            color:white;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        .topbar {{
            padding:16px 36px;
            background:linear-gradient(135deg,#0f172a,#020617);
            display:flex;
            align-items:center;
            justify-content:space-between;
            box-shadow:0 10px 24px rgba(0,0,0,0.85);
        }}
        .top-title {{
            font-size:26px;
            font-weight:800;
        }}
        .top-sub {{
            font-size:12px;
            color:#9ca3af;
        }}
        .logo-mark {{
            width:32px;
            height:32px;
            border-radius:999px;
            background:linear-gradient(135deg,#00e6e6,#007bff);
            display:flex;
            align-items:center;
            justify-content:center;
            margin-right:10px;
        }}
        .brand {{
            display:flex;
            align-items:center;
        }}
        .main {{
            padding:24px 32px 40px;
        }}
        table {{
            width:100%;
            border-collapse:collapse;
            margin-top:18px;
            font-size:13px;
        }}
        th, td {{
            padding:10px 8px;
            border-bottom:1px solid #27272f;
            text-align:center;
        }}
        th {{
            background:#020617;
            position:sticky;
            top:0;
            z-index:5;
        }}
        tr:hover td {{
            background:rgba(148,163,184,0.08);
        }}
        .back-link {{
            text-decoration:none;
            color:#00e6e6;
            font-size:13px;
        }}
    </style>
</head>
<body>
    <div class="topbar">
        <div class="brand">
            <div class="logo-mark">👁️</div>
            <div>
                <div class="top-title">Suspicious Activity Detection</div>
                <div class="top-sub">Admin Dashboard • Detailed prediction and alert logs</div>
            </div>
        </div>
        <a class="back-link" href="/">← Back to Home</a>
    </div>
    <div class="main">
        <table>
            <tr>
                <th>Time</th>
                <th>File</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>Alert</th>
            </tr>
            {rows if rows else "<tr><td colspan='5'>No records yet.</td></tr>"}
        </table>
    </div>
</body>
</html>
"""
