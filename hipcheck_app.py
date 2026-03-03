import os
import sys
import base64
import urllib.request
from io import BytesIO
import platform

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
st.set_page_config(page_title="Pose Comparison", layout="centered")

# --- sanity check for OpenCV (optional) ---
try:
    import cv2
    st.info(
        f"OpenCV imported OK • cv2={cv2.__version__} • Python={sys.version.split()[0]} • "
        f"OS={platform.system()} {platform.release()}"
    )
except Exception as e:
    st.error(
        "OpenCV failed to import. Ensure `opencv-python-headless` is in requirements.txt "
        "and **not** `opencv-python` / `opencv-contrib-python`.\n\n"
        f"Error: {e}"
    )
    # Not fatal for this app, so we don't raise

import json
from typing import Dict, Tuple, Optional

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Keep using the same import path the fork exposes
from streamlit_drawable_canvas import st_canvas  # provided by streamlit-drawable-canvas-fix

APP_TITLE = "Pose Comparison"
st.title(APP_TITLE)

# ---------------- Model download path ----------------
# Use /tmp to ensure we can write in hosted environments (e.g., Streamlit Cloud)
MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp")
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
def load_model():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.PoseLandmarker.create_from_options(options)

# ---- joints of interest ----
JOINTS = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_foot": 31, "right_foot": 32
}
LEFT_LMKS  = [11, 23, 25, 27, 31]
RIGHT_LMKS = [12, 24, 26, 28, 32]

# -------------- helpers --------------
def mp_image_from_pil(pil_img: Image.Image) -> mp.Image:
    arr = np.asarray(pil_img)  # RGB uint8
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)

def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at point b (degrees) with points given as np.array([x, y]) in normalized coords."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return np.nan
    cosine = float(np.dot(ba, bc) / denom)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def draw_line(draw: ImageDraw.ImageDraw, p1: Tuple[int, int], p2: Tuple[int, int],
              width: int = 6, fill=(255, 255, 0)) -> None:
    draw.line([p1, p2], width=width, fill=fill)

def draw_circle(draw: ImageDraw.ImageDraw, center: Tuple[int, int], radius: int,
                fill, outline=None, outline_width: int = 2) -> None:
    x, y = center
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, fill=fill, outline=outline, width=outline_width if outline else 0)

def put_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str,
             font: Optional[ImageFont.ImageFont] = None, fill=(255, 255, 255), shadow=True) -> None:
    if font is None:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
    x, y = xy
    if shadow:
        draw.text((x+1, y+1), text, font=font, fill=(30, 30, 30))
    draw.text((x, y), text, font=font, fill=fill)

def resize_for_canvas(pil_img: Image.Image, max_width: int = 900) -> Image.Image:
    """Resize to a reasonable width for canvas. Keeps aspect ratio."""
    w, h = pil_img.size
    if w <= max_width:
        return pil_img
    scale = max_width / float(w)
    return pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

def pil_to_data_url(pil_img: Image.Image) -> str:
    """Convert PIL image to base64 PNG data URL for embedding into Fabric JSON."""
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def build_initial_drawing_with_embedded_bg(pil_disp: Image.Image, joint_px: Dict[str, Tuple[int, int]],
                                           point_radius: int = 8) -> Dict:
    """
    Workaround for Streamlit Cloud background_image bugs:
    - Embed the image as a Fabric 'image' object (locked) as the first object.
    - Add draggable joint circles on top.
    """
    w, h = pil_disp.size
    data_url = pil_to_data_url(pil_disp)

    # Fabric image object (locked, non-selectable)
    bg_obj = {
        "type": "image",
        "version": "5.2.4",
        "left": 0,
        "top": 0,
        "angle": 0,
        "opacity": 1,
        "selectable": False,
        "evented": False,
        "hasControls": False,
        "hasBorders": False,
        "lockMovementX": True,
        "lockMovementY": True,
        "lockScalingX": True,
        "lockScalingY": True,
        "lockRotation": True,
        "src": data_url,
        "scaleX": 1,
        "scaleY": 1,
        "width": w,
        "height": h,
        "originX": "left",
        "originY": "top",
        "crossOrigin": "anonymous",
    }

    # Draggable circles for joints
    circles = []
    for name, (x, y) in joint_px.items():
        circles.append({
            "type": "circle",
            "left": float(x),
            "top": float(y),
            "radius": float(point_radius),
            "fill": "rgba(0, 200, 0, 0.85)",
            "stroke": "rgba(255,255,255,0.95)",
            "strokeWidth": 2,
            "hasControls": False,
            "hasBorders": False,
            "selectable": True,
            "evented": True,
            "lockScalingX": True,
            "lockScalingY": True,
            "lockRotation": True,
            # store joint id in object so we can read it back
            "name": name,
            "originX": "center",
            "originY": "center",
        })

    return {"version": "5.2.4", "objects": [bg_obj] + circles}

def extract_joint_px_from_canvas(json_data: Dict, fallback_joint_px: Dict[str, Tuple[int, int]]):
    """Read joint positions from Fabric JSON (circles)."""
    if not json_data or "objects" not in json_data:
        return dict(fallback_joint_px)

    out = dict(fallback_joint_px)
    for obj in json_data.get("objects", []):
        if obj.get("type") == "circle" and obj.get("name") in out:
            # circles are centered due to originX/Y = center
            left = obj.get("left")
            top = obj.get("top")
            if left is not None and top is not None:
                out[obj["name"]] = (int(float(left)), int(float(top)))
    return out

def joints_px_from_landmarks(landmarks, w: int, h: int):
    d = {}
    for name, idx in JOINTS.items():
        lm = landmarks[idx]
        d[name] = (int(lm.x * w), int(lm.y * h))
    return d

def normalized_from_px(px: Tuple[int, int], w: int, h: int) -> np.ndarray:
    return np.array([px[0] / float(w), px[1] / float(h)], dtype=float)

def compute_metrics_from_joint_px(joint_px: Dict[str, Tuple[int, int]], w: int, h: int,
                                  close_side_hint: Optional[str] = None) -> Dict[str, float]:
    """Compute angles using normalized coords derived from joint pixel points."""
    def P(name: str) -> np.ndarray:
        return normalized_from_px(joint_px[name], w, h)

    close_side = close_side_hint or "left"

    if close_side == "left":
        close_hip_angle = calc_angle(P("left_shoulder"), P("left_hip"), P("left_knee"))
        far_hip_angle   = calc_angle(P("right_shoulder"), P("right_hip"), P("right_knee"))
        far_knee_angle  = calc_angle(P("right_hip"), P("right_knee"), P("right_ankle"))
    else:
        close_hip_angle = calc_angle(P("right_shoulder"), P("right_hip"), P("right_knee"))
        far_hip_angle   = calc_angle(P("left_shoulder"), P("left_hip"), P("left_knee"))
        far_knee_angle  = calc_angle(P("left_hip"), P("left_knee"), P("left_ankle"))

    close_hip_flexion = 180.0 - close_hip_angle if np.isfinite(close_hip_angle) else np.nan
    far_hip_flexion   = 180.0 - far_hip_angle   if np.isfinite(far_hip_angle) else np.nan
    far_knee_extension = (far_knee_angle - 90.0) if np.isfinite(far_knee_angle) else np.nan
    jurdan_angle = (close_hip_flexion + far_knee_extension
                    if np.isfinite(close_hip_flexion) and np.isfinite(far_knee_extension) else np.nan)
    hipcheck_angle = (jurdan_angle - (90.0 - far_hip_flexion)
                      if np.isfinite(jurdan_angle) and np.isfinite(far_hip_flexion) else np.nan)

    return {
        "close_side": close_side,
        "close_hip_flexion_deg": close_hip_flexion,
        "far_hip_flexion_deg": far_hip_flexion,
        "far_knee_extension_deg": far_knee_extension,
        "jurdan_angle_deg": jurdan_angle,
        "hipcheck_angle_deg": hipcheck_angle,
    }

def annotate_with_joints(pil_img: Image.Image, joint_px: Dict[str, Tuple[int, int]], metrics: Dict[str, float]):
    """Draw skeleton + joints + metrics on a copy."""
    out = pil_img.copy()
    w, h = out.size
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    # skeleton
    for a, b in [("left_shoulder","left_hip"), ("left_hip","left_knee"),
                 ("left_knee","left_ankle"), ("left_ankle","left_foot"),
                 ("right_shoulder","right_hip"), ("right_hip","right_knee"),
                 ("right_knee","right_ankle"), ("right_ankle","right_foot")]:
        if a in joint_px and b in joint_px:
            draw_line(draw, joint_px[a], joint_px[b], width=6, fill=(255, 255, 0))

    # joints
    for name, pos in joint_px.items():
        draw_circle(draw, pos, 8, fill=(220, 0, 0), outline=(255,255,255), outline_width=2)

    # metrics overlay
    def fmt_deg(x: float) -> str:
        return f"{x:.1f}°" if (x is not None and np.isfinite(x)) else "NA"

    panel = [
        f"Close side: {metrics.get('close_side','NA')}",
        f"Close Hip Flexion: {fmt_deg(metrics.get('close_hip_flexion_deg', np.nan))}",
        f"Far Hip Flexion: {fmt_deg(metrics.get('far_hip_flexion_deg', np.nan))}",
        f"Far Knee Extension: {fmt_deg(metrics.get('far_knee_extension_deg', np.nan))}",
        f"Jurdan Angle: {fmt_deg(metrics.get('jurdan_angle_deg', np.nan))}",
        f"HipCheck Angle: {fmt_deg(metrics.get('hipcheck_angle_deg', np.nan))}",
    ]
    x0, y0 = 12, 18
    for i, line in enumerate(panel):
        put_text(draw, (x0, y0 + i*22), line, font=font)

    return out

def detect_pose_and_drag_adjust(uploaded_file, image_key: str):
    if uploaded_file is None:
        return None, None

    # Reset canvas state when a new file is uploaded
    last_key = f"last_file_{image_key}"
    if st.session_state.get(last_key) != uploaded_file.name:
        st.session_state.pop(f"init_drawing_{image_key}", None)
        st.session_state[last_key] = uploaded_file.name

    try:
        pil_full = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error(f"{image_key}: Could not open image. Please upload a valid PNG/JPG.")
        return None, None

    # --- preview (not the editable canvas) ---
    st.image(pil_full, caption=f"{image_key} preview", use_column_width=True)

    # --- detect pose on full res ---
    model = load_model()
    results = model.detect(mp_image_from_pil(pil_full))
    if not results.pose_landmarks:
        st.error(f"{image_key}: No pose detected. Try a clearer side-view photo.")
        return None, None
    landmarks = results.pose_landmarks[0]

    # --- choose close side from z (robust) ---
    def safe_mean_z(landmarks, idxs):
        vals = [getattr(landmarks[i], "z", None) for i in idxs]
        vals = [v for v in vals if v is not None]
        return float(np.nanmean(vals)) if vals else np.nan

    left_avg_z  = safe_mean_z(landmarks, LEFT_LMKS)
    right_avg_z = safe_mean_z(landmarks, RIGHT_LMKS)

    if np.isnan(left_avg_z) or np.isnan(right_avg_z):
        close_side_hint = "left"
    else:
        # In MediaPipe: smaller (often negative) z means closer to camera
        close_side_hint = "left" if left_avg_z < right_avg_z else "right"

    # --- display-size image for editing canvas (keeps interaction snappy) ---
    pil_disp = resize_for_canvas(pil_full, max_width=900)
    w_disp, h_disp = pil_disp.size

    # initial joint points in display coordinates
    joint_px0 = joints_px_from_landmarks(landmarks, w_disp, h_disp)

    st.markdown(f"### {image_key} • Drag the joint dots (live-updating angles)")

    # Build/stash initial Fabric JSON (bg image + circles)
    init_key = f"init_drawing_{image_key}"
    if init_key not in st.session_state:
        st.session_state[init_key] = build_initial_drawing_with_embedded_bg(
            pil_disp, joint_px0, point_radius=8
        )

    # Canvas
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=0,
        stroke_color="rgba(0,0,0,0)",
        background_color="rgba(0,0,0,0)",   # transparent (image is an object)
        background_image=None,              # DO NOT USE on Streamlit Cloud
        update_streamlit=True,              # live updates while dragging
        height=h_disp,
        width=w_disp,
        drawing_mode="transform",           # allows dragging objects
        initial_drawing=st.session_state[init_key],
        display_toolbar=True,
        point_display_radius=0,
        key=f"canvas_{image_key}",
    )

    # Read current positions (json_data may be None mid-drag; fallback to last init)
    json_data = canvas_result.json_data or st.session_state[init_key]
    joint_px = extract_joint_px_from_canvas(json_data, joint_px0)

    # Compute metrics live
    metrics = compute_metrics_from_joint_px(joint_px, w_disp, h_disp, close_side_hint=close_side_hint)

    # Show live values
    def fmt(x: float) -> str:
        return f"{x:.1f}°" if (x is not None and np.isfinite(x)) else "NA"

    st.markdown(
        f"""
**Live angles (drag updates immediately):**
- Close hip flexion: **{fmt(metrics['close_hip_flexion_deg'])}**
- Far hip flexion: **{fmt(metrics['far_hip_flexion_deg'])}**
- Far knee ext: **{fmt(metrics['far_knee_extension_deg'])}**
- Jurdan: **{fmt(metrics['jurdan_angle_deg'])}**
- HipCheck: **{fmt(metrics['hipcheck_angle_deg'])}**
        """
    )

    # Create annotated image for "Results"
    annot = annotate_with_joints(pil_disp, joint_px, metrics)

    # Persist the canvas JSON only when present
    if canvas_result.json_data:
        st.session_state[init_key] = canvas_result.json_data

    return annot, metrics

# ---------------- UI ----------------
st.markdown(
    "Upload two side-view photos to compare. Clear lighting and full lower-limb visibility improves results."
)

c1, c2 = st.columns(2)
with c1:
    file_a = st.file_uploader("📷 Image A", type=["jpg", "jpeg", "png"], key="upload_A")
with c2:
    file_b = st.file_uploader("📷 Image B", type=["jpg", "jpeg", "png"], key="upload_B")

annot_a, metrics_a = detect_pose_and_drag_adjust(file_a, "A")
annot_b, metrics_b = detect_pose_and_drag_adjust(file_b, "B")

if annot_a is not None or annot_b is not None:
    st.markdown("## Results (annotated overlays)")
    cols = st.columns(2)
    if annot_a is not None:
        with cols[0]:
            st.image(annot_a, caption="A • Annotated", use_column_width=True)
    if annot_b is not None:
        with cols[1]:
            st.image(annot_b, caption="B • Annotated", use_column_width=True)

    rows = []
    if metrics_a: rows.append({"Image": "A", **metrics_a})
    if metrics_b: rows.append({"Image": "B", **metrics_b})

    if rows:
        df = pd.DataFrame(rows)[
            ["Image","close_side","close_hip_flexion_deg","far_hip_flexion_deg",
             "far_knee_extension_deg","jurdan_angle_deg","hipcheck_angle_deg"]
        ]
        st.dataframe(df, use_container_width=True)

        if metrics_a and metrics_b:
            st.markdown("### A vs B (B − A)")

            def d(b, a):
                if a is None or b is None:
                    return np.nan
                if not (np.isfinite(a) and np.isfinite(b)):
                    return np.nan
                return float(b - a)

            delta_df = pd.DataFrame([{
                "Δ close_hip_flexion_deg": d(metrics_b["close_hip_flexion_deg"], metrics_a["close_hip_flexion_deg"]),
                "Δ far_hip_flexion_deg":   d(metrics_b["far_hip_flexion_deg"],   metrics_a["far_hip_flexion_deg"]),
                "Δ far_knee_extension_deg":d(metrics_b["far_knee_extension_deg"],metrics_a["far_knee_extension_deg"]),
                "Δ jurdan_angle_deg":      d(metrics_b["jurdan_angle_deg"],      metrics_a["jurdan_angle_deg"]),
                "Δ hipcheck_angle_deg":    d(metrics_b["hipcheck_angle_deg"],    metrics_a["hipcheck_angle_deg"]),
            }])
            st.dataframe(delta_df, use_container_width=True)

st.caption(
    "Angles are computed from 2D landmarks; camera angle, occlusion, and clothing can affect accuracy. "
    "Drag the points to correct them; angles update live."
)
