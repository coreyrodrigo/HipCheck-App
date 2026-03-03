import os
import base64
import urllib.request
from io import BytesIO
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

# HEIC/HEIF support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

import streamlit as st
# Wide layout so each canvas can use the full content width
st.set_page_config(page_title="Pose Comparison", layout="wide")

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Drawable canvas
from streamlit_drawable_canvas import st_canvas


# ──────────────────────────────────────────────────────────────────────────────
# App header
# ──────────────────────────────────────────────────────────────────────────────
st.title("Pose Comparison")
st.markdown(
    "- Canvas shows the **entire image** (no cropping, no distortion).\n"
    "- **Width is never limited**; we only limit **height ≈ 560 px** for usability.\n"
    "- Images are used **as-is from your phone**: we apply **EXIF orientation** only (no enhancement).\n"
    "- Upload up to **two** photos. The app auto‑assigns **Left closer** / **Right closer** from depth.\n"
    "- **Step‑wise**: Adjust keypoints → **Confirm points** → we draw the stick figure & compute **Jurdan** + **HipCheck**.\n"
    "- While you drag, **lines and angles update live** on the canvas."
)

# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe model (cached in /tmp)
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIR = "/tmp"
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_full.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
os.makedirs(MODEL_DIR, exist_ok=True)
if not os.path.exists(MODEL_PATH):
    with st.status("Downloading pose model…", expanded=False):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

@st.cache_resource
def load_pose_model():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.PoseLandmarker.create_from_options(options)


# ──────────────────────────────────────────────────────────────────────────────
# Landmarks of interest and drawing pairs
# ──────────────────────────────────────────────────────────────────────────────
JOINTS = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_foot": 31, "right_foot": 32,
}
# Fixed order used to build circles (lets us recover positions if 'name' is missing from JSON)
JOINT_ORDER: List[str] = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_foot", "right_foot",
]
# Lines for the stick figure
CONNECTED_PAIRS = [
    ("left_shoulder","left_hip"), ("left_hip","left_knee"),
    ("left_knee","left_ankle"), ("left_ankle","left_foot"),
    ("right_shoulder","right_hip"), ("right_hip","right_knee"),
    ("right_knee","right_ankle"), ("right_ankle","right_foot"),
]
LEFT_LMKS  = [11, 23, 25, 27, 31]
RIGHT_LMKS = [12, 24, 26, 28, 32]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def mp_image_from_pil(img: Image.Image) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img))

def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at point b (degrees) with points in normalized coordinates."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return np.nan
    cosang = float(np.dot(ba, bc) / denom)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def draw_line(draw: ImageDraw.ImageDraw, p1, p2, width=6, fill=(255, 255, 0)):
    draw.line([p1, p2], width=width, fill=fill)

def draw_circle(draw: ImageDraw.ImageDraw, center, radius=8, fill=(220, 0, 0),
                outline=(255, 255, 255), outline_width=2):
    x, y = center
    draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                 fill=fill, outline=outline, width=outline_width)

def put_text(draw: ImageDraw.ImageDraw, xy, text, font=None, fill=(255, 255, 255)):
    if font is None:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
    x, y = xy
    draw.text((x+1, y+1), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=fill)

def pil_to_data_url(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def add_fabric_line(x1, y1, x2, y2, color="rgba(255,255,0,0.95)", width=6):
    return {
        "type": "line",
        "x1": float(x1), "y1": float(y1),
        "x2": float(x2), "y2": float(y2),
        "stroke": color, "strokeWidth": float(width),
        "selectable": False, "evented": False,
        "hasControls": False, "hasBorders": False
    }

def add_fabric_text(x, y, text, color="#ffffff", bg="rgba(0,0,0,0.45)"):
    return {
        "type": "textbox",
        "left": float(x), "top": float(y),
        "text": str(text),
        "fontSize": 18, "fill": color,
        "backgroundColor": bg,
        "editable": False, "selectable": False,
        "evented": False, "hasControls": False, "hasBorders": False
    }

def build_fabric_scene(img_disp: Image.Image,
                       disp_joints: Dict[str, Tuple[int,int]],
                       show_lines: bool = True,
                       show_angles_panel: bool = True,
                       metrics: Optional[Dict[str, float]] = None) -> Dict:
    """
    Compose a Fabric scene: background image (locked) + optional lines + joint circles + angle panel.
    We draw lines first, then circles on top for clean handles.
    """
    w, h = img_disp.size
    bg = {
        "type": "image", "version": "5.2.4",
        "left": 0, "top": 0, "width": w, "height": h,
        "scaleX": 1, "scaleY": 1, "originX":"left","originY":"top",
        "src": pil_to_data_url(img_disp),
        "selectable": False, "evented": False,
        "hasControls": False, "hasBorders": False,
        "lockMovementX": True, "lockMovementY": True,
        "lockScalingX": True, "lockScalingY": True, "lockRotation": True
    }
    objects = [bg]

    # Optional lines (under circles so the handles are always accessible)
    if show_lines:
        for a, b in CONNECTED_PAIRS:
            x1, y1 = disp_joints[a]; x2, y2 = disp_joints[b]
            objects.append(add_fabric_line(x1, y1, x2, y2))

    # Draggable circles in fixed order
    for name in JOINT_ORDER:
        x, y = disp_joints[name]
        objects.append({
            "type":"circle","name":name,
            "left": float(x),"top": float(y),
            "radius": 8.0,
            "fill":"rgba(0,200,0,0.85)",
            "stroke":"white","strokeWidth":2,
            "selectable": True,"evented": True,
            "lockScalingX": True,"lockScalingY": True,"lockRotation": True,
            "originX":"center","originY":"center"
        })

    # Angle readout panel (top-left)
    if show_angles_panel and metrics is not None:
        def fmt(x):
            return f"{x:.1f}°" if (x is not None and np.isfinite(x)) else "NA"
        txt = (
            f"Side: {metrics.get('close_side','NA')}\n"
            f"Close Hip Flex: {fmt(metrics.get('close_hip_flexion_deg'))}\n"
            f"Far Hip Flex:   {fmt(metrics.get('far_hip_flexion_deg'))}\n"
            f"Far Knee Ext:   {fmt(metrics.get('far_knee_extension_deg'))}\n"
            f"Jurdan:         {fmt(metrics.get('jurdan_angle_deg'))}\n"
            f"HipCheck:       {fmt(metrics.get('hipcheck_angle_deg'))}"
        )
        objects.append(add_fabric_text(12, 12, txt))

    return {"version":"5.2.4","objects":objects}

def extract_display_joints_safe(json_data: Optional[Dict],
                                fallback: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
    """
    Robustly recover circle positions from Fabric JSON.
    1) Try by 'name' (ideal).
    2) If any missing, fall back to circle order we created (JOINT_ORDER).
    """
    if not json_data or "objects" not in json_data:
        return dict(fallback)

    circles = [o for o in json_data["objects"] if o.get("type") == "circle"]
    if not circles:
        return dict(fallback)

    # Try by 'name'
    out = dict(fallback)
    name_hits = 0
    for o in circles:
        nm = o.get("name")
        if nm in out:
            left = o.get("left"); top = o.get("top")
            if left is not None and top is not None:
                out[nm] = (int(float(left)), int(float(top)))
                name_hits += 1

    if name_hits == len(JOINT_ORDER):
        return out

    # Fallback to order
    out2 = dict(fallback)
    count = min(len(JOINT_ORDER), len(circles))
    for i in range(count):
        nm = JOINT_ORDER[i]
        o  = circles[i]
        left = o.get("left"); top = o.get("top")
        if left is not None and top is not None:
            out2[nm] = (int(float(left)), int(float(top)))
    return out2

def joints_from_landmarks(lmks, w: int, h: int) -> Dict[str, Tuple[int, int]]:
    out = {}
    for name, idx in JOINTS.items():
        lm = lmks[idx]
        x = int(np.clip(lm.x * w, 0, w - 1))
        y = int(np.clip(lm.y * h, 0, h - 1))
        out[name] = (x, y)
    return out

def compute_metrics(full_joints: Dict[str, Tuple[int, int]], w: int, h: int, side: str) -> Dict[str, float]:
    """
    Returns:
      close_side, close_hip_flexion_deg, far_hip_flexion_deg,
      far_knee_extension_deg, jurdan_angle_deg, hipcheck_angle_deg
    """
    def P(n):  # normalized
        x, y = full_joints[n]
        return np.array([x / w, y / h], dtype=float)

    if side == "left":
        ch = calc_angle(P("left_shoulder"), P("left_hip"), P("left_knee"))
        fh = calc_angle(P("right_shoulder"), P("right_hip"), P("right_knee"))
        fk = calc_angle(P("right_hip"), P("right_knee"), P("right_ankle"))
    else:
        ch = calc_angle(P("right_shoulder"), P("right_hip"), P("right_knee"))
        fh = calc_angle(P("left_shoulder"), P("left_hip"), P("left_knee"))
        fk = calc_angle(P("left_hip"), P("left_knee"), P("left_ankle"))

    chf = 180 - ch if np.isfinite(ch) else np.nan              # Close hip flexion
    fhf = 180 - fh if np.isfinite(fh) else np.nan              # Far hip flexion
    fke = (fk - 90) if np.isfinite(fk) else np.nan             # Far knee extension
    jurdan = (chf + fke) if np.isfinite(chf) and np.isfinite(fke) else np.nan
    hipcheck = (jurdan - (90 - fhf)) if np.isfinite(jurdan) and np.isfinite(fhf) else np.nan

    return {
        "close_side": side,
        "close_hip_flexion_deg": chf,
        "far_hip_flexion_deg": fhf,
        "far_knee_extension_deg": fke,
        "jurdan_angle_deg": jurdan,
        "hipcheck_angle_deg": hipcheck,
    }

def compute_metrics_from_display(disp_joints: Dict[str, Tuple[int,int]],
                                 disp_w: int, disp_h: int, side: str) -> Dict[str, float]:
    """Live angle math in display (scaled) coords; same formulas."""
    def P(name):
        x, y = disp_joints[name]
        return np.array([x/disp_w, y/disp_h], dtype=float)

    if side == "left":
        ch = calc_angle(P("left_shoulder"), P("left_hip"), P("left_knee"))
        fh = calc_angle(P("right_shoulder"), P("right_hip"), P("right_knee"))
        fk = calc_angle(P("right_hip"), P("right_knee"), P("right_ankle"))
    else:
        ch = calc_angle(P("right_shoulder"), P("right_hip"), P("right_knee"))
        fh = calc_angle(P("left_shoulder"), P("left_hip"), P("left_knee"))
        fk = calc_angle(P("left_hip"), P("left_knee"), P("left_ankle"))

    chf = 180 - ch if np.isfinite(ch) else np.nan
    fhf = 180 - fh if np.isfinite(fh) else np.nan
    fke = (fk - 90) if np.isfinite(fk) else np.nan
    jurdan = (chf + fke) if np.isfinite(chf) and np.isfinite(fke) else np.nan
    hipcheck = (jurdan - (90 - fhf)) if np.isfinite(jurdan) and np.isfinite(fhf) else np.nan
    return {
        "close_side": side,
        "close_hip_flexion_deg": chf,
        "far_hip_flexion_deg": fhf,
        "far_knee_extension_deg": fke,
        "jurdan_angle_deg": jurdan,
        "hipcheck_angle_deg": hipcheck,
    }

def annotate_full(img_full: Image.Image, full_joints: Dict[str, Tuple[int, int]], metrics: Dict[str, float]) -> Image.Image:
    out = img_full.copy()
    draw = ImageDraw.Draw(out)
    # skeleton
    for a, b in CONNECTED_PAIRS:
        draw_line(draw, full_joints[a], full_joints[b])
    # joints
    for pos in full_joints.values():
        draw_circle(draw, pos)
    # metrics
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    y = 20
    order = ["close_side", "close_hip_flexion_deg", "far_hip_flexion_deg",
             "far_knee_extension_deg", "jurdan_angle_deg", "hipcheck_angle_deg"]
    for k in order:
        v = metrics[k]
        text = f"{k}: {v:.1f}" if isinstance(v, float) and np.isfinite(v) else f"{k}: {v}"
        put_text(draw, (20, y), text, font)
        y += 22
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Core per-image routine (live lines/angles + step‑wise confirm)
# ──────────────────────────────────────────────────────────────────────────────
def process_image(file, label: str, index: int):
    """
    Step‑wise for each image:
      1) Adjust keypoints on canvas (lines + angles update live).
      2) Click "Confirm points" → annotate (at canvas size) & compute angles (Jurdan + HipCheck).
      3) "Edit points" re‑enables the canvas.

    Returns (confirmed: bool, side: str, disp_w: int, disp_h: int,
             annot_disp_img: PIL.Image|None, metrics: dict|None)
    """
    if file is None:
        return False, None, None, None, None, None

    # Unique per‑image id (prevents collisions even if phone reuses the same file name/size)
    uid = f"{index}-{file.name}-{getattr(file, 'size', 'na')}"
    size_key       = f"disp_size_{uid}"
    confirmed_key  = f"confirmed_{uid}"
    last_disp_key  = f"last_disp_joints_{uid}"  # persist latest dragged joints (display coords)
    metrics_key    = f"metrics_{uid}"
    annot_disp_k   = f"annot_disp_{uid}"

    # EXIF orientation ONLY (matches phone view)
    img_full = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    full_w, full_h = img_full.size

    # Compute display size ONCE per image (height cap ≈ 560 px; width follows)
    if size_key not in st.session_state:
        MAX_H = 560
        scale = min(1.0, MAX_H / float(full_h))
        disp_w = int(full_w * scale)
        disp_h = int(full_h * scale)
        st.session_state[size_key] = (disp_w, disp_h, scale)
    disp_w, disp_h, scale = st.session_state[size_key]
    img_disp = img_full if scale == 1.0 else img_full.resize((disp_w, disp_h), Image.LANCZOS)

    # Pose detection on full-resolution image (run once)
    model = load_pose_model()
    result = model.detect(mp_image_from_pil(img_full))
    if not result.pose_landmarks:
        st.error(f"{label}: Pose not detected.")
        return False, None, disp_w, disp_h, None, None
    lmks = result.pose_landmarks[0]

    # Determine close side via depth (z)
    def mean_z(indices):
        vals = [getattr(lmks[i], "z", None) for i in indices]
        vals = [v for v in vals if v is not None]
        return float(np.nanmean(vals)) if vals else np.nan
    lz = mean_z(LEFT_LMKS)
    rz = mean_z(RIGHT_LMKS)
    side = "left" if (np.isfinite(lz) and np.isfinite(rz) and lz < rz) else "right"

    # Seed joints (full‑res → display coords)
    full_joints0 = joints_from_landmarks(lmks, full_w, full_h)
    disp_joints0 = {k: (int(x * scale), int(y * scale)) for k, (x, y) in full_joints0.items()}

    st.caption(f"{label}: original {full_w}×{full_h} → canvas {disp_w}×{disp_h} (scale={scale:.3f})")

    # Step‑wise control
    confirmed = st.session_state.get(confirmed_key, False)

    if not confirmed:
        st.markdown("**Drag joints** on the canvas below. Lines + angles update live. Then click **Confirm points**.")

        # Use last dragged joints if present (smooth continuity), else detection seed
        disp_joints_seed = st.session_state.get(last_disp_key, disp_joints0)

        # Live metrics computed in display coords
        live_metrics = compute_metrics_from_display(disp_joints_seed, disp_w, disp_h, side)

        # Build a fresh scene with background + (lines) + circles + live angle panel
        live_scene = build_fabric_scene(
            img_disp, disp_joints_seed, show_lines=True, show_angles_panel=True, metrics=live_metrics
        )

        # Live updates while dragging so lines and angles move in real time
        canvas = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=0,
            background_color="rgba(0,0,0,0)",
            update_streamlit=True,           # live reruns during drag (stable sizes prevent flicker)
            height=disp_h, width=disp_w,
            drawing_mode="transform",
            initial_drawing=live_scene,      # we rebuild the scene each rerun
            display_toolbar=True,
            key=f"canvas_{uid}",
        )

        # Recover current circle positions robustly; persist for next frame & for Confirm
        json_data = canvas.json_data
        disp_joints = extract_display_joints_safe(json_data, disp_joints_seed)
        st.session_state[last_disp_key] = disp_joints  # remember latest dragged

        # Confirm button — convert latest display joints back to full-res, then annotate/compute
        if st.button(f"✅ Confirm points for {label}", key=f"confirm_{uid}"):
            disp_joints_final = st.session_state.get(last_disp_key, disp_joints)
            full_joints = {k: (int(x / scale), int(y / scale)) for k, (x, y) in disp_joints_final.items()}
            metrics = compute_metrics(full_joints, full_w, full_h, side)
            annot_full = annotate_full(img_full, full_joints, metrics)
            annot_disp = annot_full if scale == 1.0 else annot_full.resize((disp_w, disp_h), Image.LANCZOS)

            # Persist for results
            st.session_state[confirmed_key] = True
            st.session_state[metrics_key]   = metrics
            st.session_state[annot_disp_k]  = annot_disp

            confirmed = True

    else:
        # Already confirmed → show the annotated image at the SAME SIZE as the canvas
        annot_disp = st.session_state.get(annot_disp_k)
        if annot_disp is not None:
            st.markdown("**Confirmed:** Pose drawn & angles calculated. Click **Edit points** to tweak.")
            st.image(np.array(annot_disp), width=disp_w)  # exact canvas size
            if st.button(f"✏️ Edit points for {label}", key=f"edit_{uid}"):
                st.session_state[confirmed_key] = False
                confirmed = False

    # Return current state for collector
    if st.session_state.get(confirmed_key, False):
        return True, side, disp_w, disp_h, st.session_state.get(annot_disp_k), st.session_state.get(metrics_key)
    else:
        return False, side, disp_w, disp_h, None, None


# ──────────────────────────────────────────────────────────────────────────────
# UI: upload & canvases (tabs → full width per image; both always shown)
# ──────────────────────────────────────────────────────────────────────────────
files = st.file_uploader(
    "Upload up to 2 images (JPG/PNG/HEIC). Phone orientation is preserved; canvas height ≈ 560 px.",
    type=["jpg", "jpeg", "png", "heic", "HEIC", "heif", "HEIF"],
    accept_multiple_files=True
)

if files and len(files) > 2:
    st.warning("You uploaded more than two images; using the first two.")
    files = files[:2]

# Collect per‑image results so Image 2 always shows — even if both are the same side
per_image_results: List[Optional[Tuple[Image.Image, dict, str, int, int]]] = [None] * (len(files) if files else 0)

if files:
    tab_labels = [f"Image {i+1}" for i in range(len(files))]
    tabs = st.tabs(tab_labels)
    for idx, (t, f) in enumerate(zip(tabs, files), start=1):
        with t:
            confirmed, side, dw, dh, annot_disp, metrics = process_image(f, f"Image {idx}", idx)
            if confirmed and annot_disp is not None and metrics is not None:
                per_image_results[idx-1] = (annot_disp, metrics, side, dw, dh)

# ──────────────────────────────────────────────────────────────────────────────
# Results (each image shown at exactly the same size as its canvas)
# ──────────────────────────────────────────────────────────────────────────────
if any(per_image_results):
    st.header("Results (Annotated Pose at Canvas Size)")

    # One tab per image so both always render
    r_tabs = st.tabs([f"Image {i+1}" for i in range(len(per_image_results))])
    for i, rt in enumerate(r_tabs):
        with rt:
            if per_image_results[i] is None:
                st.info("Not confirmed yet.")
                continue
            annot_disp, metrics, side, dw, dh = per_image_results[i]
            st.subheader(f"{side.capitalize()} closer")
            st.image(np.array(annot_disp), width=dw)  # exact canvas size

    # Metrics table (only confirmed images)
    rows = []
    for i, pack in enumerate(per_image_results, start=1):
        if pack:
            _, met, side, _, _ = pack
            rows.append({"Image": f"Image {i}", **met})
    if rows:
        st.subheader("Metrics (includes Jurdan & HipCheck)")
        cols = ["Image", "close_side", "close_hip_flexion_deg", "far_hip_flexion_deg",
                "far_knee_extension_deg", "jurdan_angle_deg", "hipcheck_angle_deg"]
        df = pd.DataFrame(rows)[cols]
        st.dataframe(df)

        # If we have one left and one right, show Right − Left deltas
        left_row  = next((r for r in rows if r["close_side"] == "left"), None)
        right_row = next((r for r in rows if r["close_side"] == "right"), None)
        if left_row and right_row:
            st.subheader("Right − Left (Δ)")
            def d(b, a):
                if a is None or b is None: return np.nan
                if not (np.isfinite(a) and np.isfinite(b)): return np.nan
                return float(b - a)
            delta = pd.DataFrame([{
                "Δ close_hip_flexion_deg":  d(right_row["close_hip_flexion_deg"],  left_row["close_hip_flexion_deg"]),
                "Δ far_hip_flexion_deg":    d(right_row["far_hip_flexion_deg"],    left_row["far_hip_flexion_deg"]),
                "Δ far_knee_extension_deg": d(right_row["far_knee_extension_deg"], left_row["far_knee_extension_deg"]),
                "Δ jurdan_angle_deg":       d(right_row["jurdan_angle_deg"],       left_row["jurdan_angle_deg"]),
                "Δ hipcheck_angle_deg":     d(right_row["hipcheck_angle_deg"],     left_row["hipcheck_angle_deg"]),
            }])
            st.dataframe(delta)

st.caption(
    "Adjust → Confirm → stick figure & angles. Live lines/angles on the canvas while dragging. "
    "Canvas shows the full photo; width never limited; height ≈ 560 px; EXIF orientation only; "
    "no cropping or enhancement; no 'use_container_width' used."
)
