import streamlit as st

st.set_page_config(page_title="Pose Comparison", layout="centered")

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request
from io import BytesIO

APP_TITLE = "Pose Comparison"
st.title(APP_TITLE)

# -------- Model download path (no __file__ dependency for Streamlit) --------
MODEL_PATH = os.path.join(os.getcwd(), "pose_landmarker_full.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

if not os.path.exists(MODEL_PATH):
    with st.status("Downloading pose model…", expanded=False):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

@st.cache_resource
def load_model():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        running_mode=vision.RunningMode.IMAGE
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

def mp_image_from_pil(pil_img: Image.Image) -> mp.Image:
    """Create a MediaPipe Image from a PIL RGB image (no temp files, no cv2)."""
    arr = np.asarray(pil_img)  # shape (H, W, 3), uint8, RGB
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)

def calc_angle(a, b, c):
    """Angle at point b (degrees) with points given as np.array([x, y])."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return np.nan
    cosine = float(np.dot(ba, bc) / denom)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def draw_line(draw: ImageDraw.ImageDraw, p1, p2, width=6, fill=(255, 255, 0)):
    draw.line([p1, p2], width=width, fill=fill)

def draw_circle(draw: ImageDraw.ImageDraw, center, radius, fill, outline=None, outline_width=2):
    x, y = center
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, fill=fill, outline=outline, width=outline_width if outline else 0)

def put_text(draw: ImageDraw.ImageDraw, xy, text, font=None, fill=(255, 255, 255), shadow=True):
    if font is None:
        font = ImageFont.load_default()
    x, y = xy
    if shadow:
        draw.text((x+1, y+1), text, font=font, fill=(30, 30, 30))
    draw.text((x, y), text, font=font, fill=fill)

def create_adjustment_interface(pil_image, landmarks, width, height, image_key):
    """Form UI to optionally adjust joint pixel coordinates; persists to session_state."""
    # Preview with AI points
    preview = pil_image.copy()
    d = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    for name, idx in JOINTS.items():
        pos = (int(landmarks[idx].x * width), int(landmarks[idx].y * height))
        draw_circle(d, pos, 6, fill=(0, 200, 0))
        put_text(d, (pos[0] + 8, pos[1] - 8), name.replace('_', ' '), font=font)

    st.image(preview, caption=f"{image_key}: AI Detected Joints (green). Adjust below if needed.", use_container_width=True)

    st.markdown(f"#### {image_key} • Manual Joint Position Adjustment")
    st.caption("Adjust only if a joint is clearly misplaced. Values are in **pixels**.")
    with st.form(f"adjust_joints_{image_key}"):
        adjustments = {}

        left_exp = st.expander("🔧 Adjust Left Side", expanded=False)
        with left_exp:
            for joint_name in ["left_shoulder", "left_hip", "left_knee", "left_ankle", "left_foot"]:
                idx = JOINTS[joint_name]
                ox = int(landmarks[idx].x * width)
                oy = int(landmarks[idx].y * height)
                st.markdown(f"**{joint_name.replace('_',' ').title()}** _(original: {ox}, {oy})_")
                c1, c2, c3 = st.columns([2,2,1])
                with c1:
                    nx = st.number_input("X", min_value=0, max_value=int(width), value=int(ox),
                                         key=f"{joint_name}_x_{image_key}")
                with c2:
                    ny = st.number_input("Y", min_value=0, max_value=int(height), value=int(oy),
                                         key=f"{joint_name}_y_{image_key}")
                with c3:
                    changed = (nx != ox) or (ny != oy)
                    st.write("✏️" if changed else "📍")
                if changed:
                    adjustments[joint_name] = {
                        "x": int(nx), "y": int(ny),
                        "normalized_x": float(nx / width),
                        "normalized_y": float(ny / height),
                    }

        right_exp = st.expander("🔧 Adjust Right Side", expanded=False)
        with right_exp:
            for joint_name in ["right_shoulder", "right_hip", "right_knee", "right_ankle", "right_foot"]:
                idx = JOINTS[joint_name]
                ox = int(landmarks[idx].x * width)
                oy = int(landmarks[idx].y * height)
                st.markdown(f"**{joint_name.replace('_',' ').title()}** _(original: {ox}, {oy})_")
                c1, c2, c3 = st.columns([2,2,1])
                with c1:
                    nx = st.number_input("X", min_value=0, max_value=int(width), value=int(ox),
                                         key=f"{joint_name}_x_{image_key}")
                with c2:
                    ny = st.number_input("Y", min_value=0, max_value=int(height), value=int(oy),
                                         key=f"{joint_name}_y_{image_key}")
                with c3:
                    changed = (nx != ox) or (ny != oy)
                    st.write("✏️" if changed else "📍")
                if changed:
                    adjustments[joint_name] = {
                        "x": int(nx), "y": int(ny),
                        "normalized_x": float(nx / width),
                        "normalized_y": float(ny / height),
                    }

        submitted = st.form_submit_button("🔄 Apply Adjustments & Recalculate", type="primary")
        if submitted:
            st.session_state[f"adjustments_{image_key}"] = adjustments
            if adjustments:
                st.success(f"Applied {len(adjustments)} joint adjustment(s) for {image_key}.")
            else:
                st.info("No adjustments made; using AI predictions.")
            st.rerun()

    return st.session_state.get(f"adjustments_{image_key}", {})

def apply_adjustments_to_landmarks(landmarks, adjustments):
    """Return a new landmark list with overridden normalized x/y for adjusted joints."""
    if not adjustments:
        return landmarks

    class AdjustedLandmark:
        def __init__(self, x, y, z, visibility, presence):
            self.x = x
            self.y = y
            self.z = z
            self.visibility = visibility
            self.presence = presence

    adjusted = []
    for i, lm in enumerate(landmarks):
        name = next((n for n, idx in JOINTS.items() if idx == i), None)
        if name and name in adjustments:
            adj = adjustments[name]
            adjusted.append(
                AdjustedLandmark(
                    x=adj["normalized_x"],
                    y=adj["normalized_y"],
                    z=lm.z,
                    visibility=getattr(lm, "visibility", 1.0),
                    presence=getattr(lm, "presence", 1.0),
                )
            )
        else:
            adjusted.append(lm)
    return adjusted

def annotate_and_compute(pil_image: Image.Image, landmarks, manual_adjustments=None):
    """Compute metrics & draw over a copy of the image. Returns (annotated PIL image, metrics dict)."""
    w, h = pil_image.size
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()

    def to_px(idx):
        lm = landmarks[idx]
        return (int(lm.x * w), int(lm.y * h))

    def get_xy(idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y], dtype=float)

    # determine close side by average z (smaller z ~ closer)
    left_avg_z = np.mean([landmarks[i].z for i in LEFT_LMKS])
    right_avg_z = np.mean([landmarks[i].z for i in RIGHT_LMKS])
    close_side = "left" if left_avg_z < right_avg_z else "right"

    if close_side == "left":
        close_hip_angle = calc_angle(get_xy(11), get_xy(23), get_xy(25))
        far_hip_angle   = calc_angle(get_xy(12), get_xy(24), get_xy(26))
        far_knee_angle  = calc_angle(get_xy(24), get_xy(26), get_xy(28))
    else:
        close_hip_angle = calc_angle(get_xy(12), get_xy(24), get_xy(26))
        far_hip_angle   = calc_angle(get_xy(11), get_xy(23), get_xy(25))
        far_knee_angle  = calc_angle(get_xy(23), get_xy(25), get_xy(27))

    close_hip_flexion = 180.0 - close_hip_angle if np.isfinite(close_hip_angle) else np.nan
    far_hip_flexion   = 180.0 - far_hip_angle   if np.isfinite(far_hip_angle) else np.nan
    far_knee_extension = (far_knee_angle - 90.0) if np.isfinite(far_knee_angle) else np.nan
    jurdan_angle = (close_hip_flexion + far_knee_extension
                    if np.isfinite(close_hip_flexion) and np.isfinite(far_knee_extension) else np.nan)
    hipcheck_angle = (jurdan_angle - (90.0 - far_hip_flexion)
                      if np.isfinite(jurdan_angle) and np.isfinite(far_hip_flexion) else np.nan)

    # skeleton lines
    for a, b in [("left_shoulder","left_hip"), ("left_hip","left_knee"),
                 ("left_knee","left_ankle"), ("left_ankle","left_foot"),
                 ("right_shoulder","right_hip"), ("right_hip","right_knee"),
                 ("right_knee","right_ankle"), ("right_ankle","right_foot")]:
        draw_line(draw, to_px(JOINTS[a]), to_px(JOINTS[b]), width=6, fill=(255, 255, 0))

    # joints: orange if adjusted, red otherwise
    for name, idx in JOINTS.items():
        pos = to_px(idx)
        if manual_adjustments and name in manual_adjustments:
            draw_circle(draw, pos, 10, fill=(255, 140, 0), outline=(255,255,255), outline_width=3)
            put_text(draw, (pos[0]-5, pos[1]+8), "M", font=font)
        else:
            draw_circle(draw, pos, 8, fill=(220, 0, 0), outline=(255,255,255), outline_width=2)

    # metrics panel
    panel = [
        f"Close side: {close_side}",
        f"Close Hip Flexion: {close_hip_flexion:.1f}°" if np.isfinite(close_hip_flexion) else "Close Hip Flexion: NA",
        f"Far Hip Flexion: {far_hip_flexion:.1f}°"     if np.isfinite(far_hip_flexion) else "Far Hip Flexion: NA",
        f"Far Knee Extension: {far_knee_extension:.1f}°" if np.isfinite(far_knee_extension) else "Far Knee Extension: NA",
        f"Jurdan Angle: {jurdan_angle:.1f}°" if np.isfinite(jurdan_angle) else "Jurdan Angle: NA",
        f"HipCheck Angle: {hipcheck_angle:.1f}°" if np.isfinite(hipcheck_angle) else "HipCheck Angle: NA",
    ]
    x0, y0 = 12, 18
    for i, line in enumerate(panel):
        put_text(draw, (x0, y0 + i*20), line, font=font)

    metrics = {
        "close_side": close_side,
        "close_hip_flexion_deg": close_hip_flexion,
        "far_hip_flexion_deg": far_hip_flexion,
        "far_knee_extension_deg": far_knee_extension,
        "jurdan_angle_deg": jurdan_angle,
        "hipcheck_angle_deg": hipcheck_angle,
    }
    return pil_image, metrics

def detect_pose_and_optionally_adjust(uploaded_file, image_key: str):
    """Full pipeline for one image using only PIL + MediaPipe."""
    if uploaded_file is None:
        return None, None
    try:
        pil_img = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error(f"{image_key}: Could not open image. Please upload a valid PNG/JPG.")
        return None, None

    w, h = pil_img.size
    mp_img = mp_image_from_pil(pil_img)
    model = load_model()
    results = model.detect(mp_img)

    if not results.pose_landmarks:
        st.error(f"{image_key}: No pose detected. Try a clearer side-view photo.")
        return None, None

    landmarks = results.pose_landmarks[0]
    # UI for adjustments
    adjustments = create_adjustment_interface(pil_img, landmarks, w, h, image_key)
    # apply & recompute on a fresh copy for drawing
    adjusted_landmarks = apply_adjustments_to_landmarks(landmarks, adjustments)
    annotated, metrics = annotate_and_compute(pil_img.copy(), adjusted_landmarks, adjustments)
    return annotated, metrics

# ---------------- UI ----------------
st.markdown("Upload two side-view photos to compare. Clear lighting and full lower-limb visibility improves results.")

c1, c2 = st.columns(2)
with c1:
    file_a = st.file_uploader("📷 Image A", type=["jpg", "jpeg", "png"], key="upload_A")
with c2:
    file_b = st.file_uploader("📷 Image B", type=["jpg", "jpeg", "png"], key="upload_B")

annot_a, metrics_a = detect_pose_and_optionally_adjust(file_a, "A")
annot_b, metrics_b = detect_pose_and_optionally_adjust(file_b, "B")

if annot_a is not None or annot_b is not None:
    st.markdown("### Results")
    cols = st.columns(2)
    if annot_a is not None:
        with cols[0]:
            st.image(annot_a, caption="A • Annotated", use_container_width=True)
    if annot_b is not None:
        with cols[1]:
            st.image(annot_b, caption="B • Annotated", use_container_width=True)

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
            st.markdown("#### A vs B (B − A)")
            def d(b,a):
                if any(x is None for x in (a,b)): return np.nan
                if not (np.isfinite(a) and np.isfinite(b)): return np.nan
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
    "Manual joint correction overrides only the selected points (normalized to image size)."
)
