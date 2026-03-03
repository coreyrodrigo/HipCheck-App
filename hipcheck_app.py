import os
import base64
import urllib.request
from io import BytesIO
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# HEIC/HEIF support (no auto-rotation or enhancement)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

import streamlit as st
st.set_page_config(page_title="Pose Comparison", layout="centered")

# Optional OpenCV (not required for this minimal flow)
try:
    import cv2  # noqa: F401
except Exception:
    pass

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Drawable canvas
from streamlit_drawable_canvas import st_canvas

# -------------------------------------------------------------------
# App title & brief instructions
# -------------------------------------------------------------------
st.title("Pose Comparison")
st.markdown(
    "This version **displays the full photo** (no cropping, no distortion), "
    "scaled to a max **900 px width** for speed and usability. "
    "No auto-rotation and no enhancement are applied. Upload up to **two** images."
)

# -------------------------------------------------------------------
# MediaPipe Pose model: cached & stored in /tmp (Cloud-safe)
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Landmarks of interest
# -------------------------------------------------------------------
JOINTS = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_foot": 31, "right_foot": 32,
}
LEFT_LMKS  = [11, 23, 25, 27, 31]
RIGHT_LMKS = [12, 24, 26, 28, 32]

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
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

def build_canvas_json(display_img: Image.Image, display_joints: Dict[str, Tuple[int, int]]) -> Dict:
    """Fabric JSON with embedded image + draggable joint circles, all in display (scaled) coords."""
    w, h = display_img.size
    bg = {
      "type": "image", "version": "5.2.4",
      "left": 0, "top": 0, "width": w, "height": h,
      "scaleX": 1, "scaleY": 1, "originX": "left", "originY": "top",
      "src": pil_to_data_url(display_img),
      "selectable": False, "evented": False,
      "hasControls": False, "hasBorders": False,
      "lockMovementX": True, "lockMovementY": True,
      "lockScalingX": True, "lockScalingY": True, "lockRotation": True
    }
    circles = []
    for name, (x, y) in display_joints.items():
        circles.append({
            "type": "circle", "name": name,
            "left": float(x), "top": float(y),
            "radius": 8.0,
            "fill": "rgba(0,200,0,0.85)",
            "stroke": "white", "strokeWidth": 2,
            "selectable": True, "evented": True,
            "lockScalingX": True, "lockScalingY": True, "lockRotation": True,
            "originX": "center", "originY": "center"
        })
    return {"version": "5.2.4", "objects": [bg] + circles}

def extract_display_joints(json_data: Optional[Dict], fallback: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
    if not json_data or "objects" not in json_data:
        return dict(fallback)
    out = dict(fallback)
    for obj in json_data["objects"]:
        if obj.get("type") == "circle":
            name = obj.get("name")
            if name in out:
                left = obj.get("left")
                top  = obj.get("top")
                if left is not None and top is not None:
                    out[name] = (int(float(left)), int(float(top)))
    return out

def joints_from_landmarks(lmks, w: int, h: int) -> Dict[str, Tuple[int, int]]:
    out = {}
    for name, idx in JOINTS.items():
        lm = lmks[idx]
        x = int(np.clip(lm.x * w, 0, w - 1))
        y = int(np.clip(lm.y * h, 0, h - 1))
        out[name] = (x, y)
    return out

def compute_metrics(full_joints: Dict[str, Tuple[int, int]], w: int, h: int, side: str) -> Dict[str, float]:
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
    for a, b in [
        ("left_shoulder","left_hip"), ("left_hip","left_knee"),
        ("left_knee","left_ankle"), ("left_ankle","left_foot"),
        ("right_shoulder","right_hip"), ("right_hip","right_knee"),
        ("right_knee","right_ankle"), ("right_ankle","right_foot")
    ]:
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
    for k in ["close_side", "close_hip_flexion_deg", "far_hip_flexion_deg",
              "far_knee_extension_deg", "jurdan_angle_deg", "hipcheck_angle_deg"]:
        v = metrics[k]
        text = f"{k}: {v:.1f}" if isinstance(v, float) and np.isfinite(v) else f"{k}: {v}"
        put_text(draw, (20, y), text, font)
        y += 22
    return out

# -------------------------------------------------------------------
# Core per-image routine
# -------------------------------------------------------------------
def process_image(file, label: str):
    """
    - Load as-is (no EXIF rotation, no enhancement).
    - Display-scaled to max 900px width (keep aspect; entire image visible).
    - Canvas runs on scaled image; joints mapped back to full resolution.
    - Returns (annotated_fullres, metrics, side).
    """
    if file is None:
        return None, None, None

    # Load EXACTLY as provided (no EXIF rotate, no enhancement)
    img_full = Image.open(file).convert("RGB")
    full_w, full_h = img_full.size

    # --- display scaling (Option A) ---
    MAX_W = 900
    scale = min(1.0, MAX_W / float(full_w))
    disp_w = int(full_w * scale)
    disp_h = int(full_h * scale)
    img_disp = img_full if scale == 1.0 else img_full.resize((disp_w, disp_h), Image.LANCZOS)

    # Show the image (Streamlit will fit width; we also show exact size in text)
    st.caption(f"{label}: {full_w}×{full_h} → display {disp_w}×{disp_h} (scale={scale:.3f})")
    st.image(img_full, caption=f"{label} (original, full image shown below on canvas at {disp_w}×{disp_h})",
             use_container_width=False)

    # Pose detection on the full-resolution image for accuracy
    model = load_pose_model()
    result = model.detect(mp_image_from_pil(img_full))
    if not result.pose_landmarks:
        st.error(f"{label}: Pose not detected.")
        return None, None, None
    lmks = result.pose_landmarks[0]

    # Decide which side is closer (z-depth)
    def mean_z(indices):
        vals = [getattr(lmks[i], "z", None) for i in indices]
        vals = [v for v in vals if v is not None]
        return float(np.nanmean(vals)) if vals else np.nan
    lz, rz = mean_z(LEFT_LMKS), mean_z(RIGHT_LMKS)
    side = "left" if (np.isfinite(lz) and np.isfinite(rz) and lz < rz) else "right"

    # Full-res joints
    full_joints0 = joints_from_landmarks(lmks, full_w, full_h)
    # Scaled display joints
    disp_joints0 = {k: (int(x * scale), int(y * scale)) for k, (x, y) in full_joints0.items()}

    st.markdown(f"### {label} – Drag joints (canvas shown at {disp_w}×{disp_h}, no cropping)")
    ss_key = f"init_json_{label}_{file.name}"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = build_canvas_json(img_disp, disp_joints0)

    canvas = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=0,
        background_color="rgba(0,0,0,0)",
        update_streamlit=True,
        height=disp_h,
        width=disp_w,
        drawing_mode="transform",
        initial_drawing=st.session_state[ss_key],
        display_toolbar=True,
        key=f"canvas_{label}_{file.name}",
    )

    json_data = canvas.json_data or st.session_state[ss_key]
    disp_joints = extract_display_joints(json_data, disp_joints0)

    # Map back to full resolution for metric computation & annotation
    full_joints = {k: (int(x / scale), int(y / scale)) for k, (x, y) in disp_joints.items()}

    metrics = compute_metrics(full_joints, full_w, full_h, side)
    annot_full = annotate_full(img_full, full_joints, metrics)

    if canvas.json_data:
        st.session_state[ss_key] = canvas.json_data

    return annot_full, metrics, side

# -------------------------------------------------------------------
# UI: upload & results
# -------------------------------------------------------------------
files = st.file_uploader(
    "Upload up to 2 images (JPG/PNG/HEIC) — images are used as-is; the canvas is scaled to 900 px width max.",
    type=["jpg", "jpeg", "png", "heic", "HEIC", "heif", "HEIF"],
    accept_multiple_files=True
)

if files and len(files) > 2:
    st.warning("You uploaded more than two images; using the first two.")
    files = files[:2]

left_pack, right_pack = None, None

if files:
    for i, f in enumerate(files, 1):
        with st.container(border=True):
            annot, metrics, side = process_image(f, f"Image {i}")
            if annot is None:
                continue
            if side == "left" and left_pack is None:
                left_pack = (annot, metrics)
            elif side == "right" and right_pack is None:
                right_pack = (annot, metrics)
            else:
                st.info(f"Image {i} also appears to be **{side}-closer**. Capture the opposite side for comparison.")

# Results
if left_pack or right_pack:
    st.header("Results (Annotated Full‑Res Outputs)")
    cols = st.columns(2)
    if left_pack:
        with cols[0]:
            st.subheader("Left Closer")
            st.image(left_pack[0], use_container_width=True)
    if right_pack:
        with cols[1]:
            st.subheader("Right Closer")
            st.image(right_pack[0], use_container_width=True)

    rows = []
    if left_pack:  rows.append({"Image": "Left closer",  **left_pack[1]})
    if right_pack: rows.append({"Image": "Right closer", **right_pack[1]})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if left_pack and right_pack:
            st.subheader("Right − Left (Δ)")
            def d(b, a):
                if a is None or b is None:
                    return np.nan
                if not (np.isfinite(a) and np.isfinite(b)):
                    return np.nan
                return float(b - a)
            L, R = left_pack[1], right_pack[1]
            delta = pd.DataFrame([{
                "Δ close_hip_flexion_deg":  d(R["close_hip_flexion_deg"],  L["close_hip_flexion_deg"]),
                "Δ far_hip_flexion_deg":    d(R["far_hip_flexion_deg"],    L["far_hip_flexion_deg"]),
                "Δ far_knee_extension_deg": d(R["far_knee_extension_deg"], L["far_knee_extension_deg"]),
                "Δ jurdan_angle_deg":       d(R["jurdan_angle_deg"],       L["jurdan_angle_deg"]),
                "Δ hipcheck_angle_deg":     d(R["hipcheck_angle_deg"],     L["hipcheck_angle_deg"]),
            }])
            st.dataframe(delta, use_container_width=True)

st.caption("Full image is always shown on the canvas (scaled uniformly to ≤900 px width); no cropping; aspect ratio preserved.")
