import streamlit as st

st.set_page_config(page_title="Pose Comparison", layout="centered")

import numpy as np
import cv2
import pandas as pd
from PIL import Image
from datetime import datetime
import tempfile
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request
from io import BytesIO

# =========================
# Constants & Setup
# =========================
APP_TITLE = "Pose Comparison"
st.title(APP_TITLE)

# Safely place the model in current working directory (works in Streamlit)
MODEL_PATH = os.path.join(os.getcwd(), "pose_landmarker_full.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

# Download once if missing
if not os.path.exists(MODEL_PATH):
    with st.status("Downloading pose model…", expanded=False):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# MediaPipe setup (Tasks API)
@st.cache_resource
def load_model():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        running_mode=vision.RunningMode.IMAGE
    )
    return vision.PoseLandmarker.create_from_options(options)

# Landmark indices of interest
JOINTS = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_foot": 31, "right_foot": 32
}
LEFT_LMKS = [11, 23, 25, 27, 31]
RIGHT_LMKS = [12, 24, 26, 28, 32]

# =========================
# Utility Functions
# =========================
def mp_image_from_filelike(file_like) -> mp.Image:
    """Read any uploaded file-like into a real temp file for MP Image."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(file_like.read())
        tmp_path = tmp.name
    mp_img = mp.Image.create_from_file(tmp_path)
    # also read with cv2 for drawing/display
    image_bgr = cv2.imread(tmp_path)
    os.remove(tmp_path)  # cleanup temp
    return mp_img, image_bgr

def calc_angle(a, b, c):
    """
    Returns the smaller interior angle at point b (in degrees) formed by points a-b-c.
    a, b, c are np.array([x, y]) in normalized or pixel space (consistent within call).
    """
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return np.nan
    cosine = float(np.dot(ba, bc) / denom)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def create_adjustment_interface(image_bgr, landmarks, width, height, image_key):
    """
    Form-based interface for adjusting selected joint positions.
    Stores adjustments in st.session_state[f"adjustments_{image_key}"].
    Returns the current adjustments dict.
    """
    # Draw the detected joints for context
    display_img = image_bgr.copy()
    for name, idx in JOINTS.items():
        pos = (int(landmarks[idx].x * width), int(landmarks[idx].y * height))
        cv2.circle(display_img, pos, 8, (0, 255, 0), -1)  # green for AI
        cv2.putText(display_img, name.replace('_', ' '), (pos[0] + 10, pos[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB),
             caption=f"{image_key}: AI Detected Joints (green). Adjust below if needed.",
             use_container_width=True)

    st.markdown(f"#### {image_key} • Manual Joint Position Adjustment")
    st.caption("Adjust only if a joint is clearly misplaced. Values are in **pixels**.")
    with st.form(f"adjust_joints_{image_key}"):
        adjustments = {}
        left_expander = st.expander("🔧 Adjust Left Side", expanded=False)
        with left_expander:
            for joint_name in ["left_shoulder", "left_hip", "left_knee", "left_ankle", "left_foot"]:
                idx = JOINTS[joint_name]
                ox = int(landmarks[idx].x * width)
                oy = int(landmarks[idx].y * height)
                st.markdown(f"**{joint_name.replace('_', ' ').title()}**  "
                            f"_(original: {ox}, {oy})_")
                c1, c2, c3 = st.columns([2, 2, 1])
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
                        "normalized_y": float(ny / height)
                    }

        right_expander = st.expander("🔧 Adjust Right Side", expanded=False)
        with right_expander:
            for joint_name in ["right_shoulder", "right_hip", "right_knee", "right_ankle", "right_foot"]:
                idx = JOINTS[joint_name]
                ox = int(landmarks[idx].x * width)
                oy = int(landmarks[idx].y * height)
                st.markdown(f"**{joint_name.replace('_', ' ').title()}**  "
                            f"_(original: {ox}, {oy})_")
                c1, c2, c3 = st.columns([2, 2, 1])
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
                        "normalized_y": float(ny / height)
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
    """
    Return a new landmark list where adjusted joints have overridden x/y (normalized).
    """
    if not adjustments:
        return landmarks

    adjusted = []
    # Create a tiny carrier for adjusted values with same attrs MediaPipe expects
    class AdjustedLandmark:
        def __init__(self, x, y, z, visibility, presence):
            self.x = x
            self.y = y
            self.z = z
            self.visibility = visibility
            self.presence = presence

    # Iterate all landmarks; replace only those we care about
    for i, lm in enumerate(landmarks):
        joint_name = None
        for name, idx in JOINTS.items():
            if idx == i:
                joint_name = name
                break
        if joint_name and joint_name in adjustments:
            adj = adjustments[joint_name]
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

def annotate_and_compute(image_bgr, landmarks, manual_adjustments=None):
    """
    Compute angles/metrics & draw an annotated overlay.
    Returns (annotated_bgr, metrics_dict).
    """
    h, w = image_bgr.shape[:2]

    # Helper functions bound to current dimensions/landmarks
    def to_px(idx):
        lm = landmarks[idx]
        return (int(lm.x * w), int(lm.y * h))

    def get_xy(idx):
        lm = landmarks[idx]
        return np.array([lm.x, lm.y], dtype=float)

    # Determine close/far side using average z (smaller z ~ closer to camera)
    left_avg_z = np.mean([landmarks[i].z for i in LEFT_LMKS])
    right_avg_z = np.mean([landmarks[i].z for i in RIGHT_LMKS])
    close_side = "left" if left_avg_z < right_avg_z else "right"

    # Compute angles based on close/far definition
    if close_side == "left":
        close_hip_angle = calc_angle(get_xy(11), get_xy(23), get_xy(25))
        far_hip_angle   = calc_angle(get_xy(12), get_xy(24), get_xy(26))
        far_knee_angle  = calc_angle(get_xy(24), get_xy(26), get_xy(28))
    else:
        close_hip_angle = calc_angle(get_xy(12), get_xy(24), get_xy(26))
        far_hip_angle   = calc_angle(get_xy(11), get_xy(23), get_xy(25))
        far_knee_angle  = calc_angle(get_xy(23), get_xy(25), get_xy(27))

    # Convert to flex/extension style metrics
    close_hip_flexion = 180.0 - close_hip_angle if np.isfinite(close_hip_angle) else np.nan
    far_hip_flexion   = 180.0 - far_hip_angle   if np.isfinite(far_hip_angle) else np.nan
    # Far knee extension as (angle - 90) per your spec
    far_knee_extension = (far_knee_angle - 90.0) if np.isfinite(far_knee_angle) else np.nan

    jurdan_angle = (close_hip_flexion + far_knee_extension
                    if np.isfinite(close_hip_flexion) and np.isfinite(far_knee_extension)
                    else np.nan)
    # HipCheck = Jurdan - (90 - far_hip_flexion)
    hipcheck_angle = (jurdan_angle - (90.0 - far_hip_flexion)
                      if np.isfinite(jurdan_angle) and np.isfinite(far_hip_flexion)
                      else np.nan)

    # Draw skeleton-ish lines
    draw = image_bgr.copy()
    def draw_line(a, b):
        cv2.line(draw, to_px(JOINTS[a]), to_px(JOINTS[b]), (0, 255, 255), 6)

    for pair in [("left_shoulder", "left_hip"),
                 ("left_hip", "left_knee"),
                 ("left_knee", "left_ankle"),
                 ("left_ankle", "left_foot"),
                 ("right_shoulder", "right_hip"),
                 ("right_hip", "right_knee"),
                 ("right_knee", "right_ankle"),
                 ("right_ankle", "right_foot")]:
        draw_line(*pair)

    # Joints: mark adjusted vs AI
    for name, idx in JOINTS.items():
        pos = to_px(idx)
        if manual_adjustments and name in manual_adjustments:
            # Orange for manually adjusted
            cv2.circle(draw, pos, 12, (0, 165, 255), -1)
            cv2.circle(draw, pos, 12, (255, 255, 255), 3)
            cv2.putText(draw, "M", (pos[0] - 5, pos[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        else:
            # Red for AI-detected
            cv2.circle(draw, pos, 10, (0, 0, 255), -1)
            cv2.circle(draw, pos, 10, (255, 255, 255), 2)

    # Put a small panel of metrics on the image
    panel = [
        f"Close side: {close_side}",
        f"Close Hip Flexion: {close_hip_flexion:.1f} deg" if np.isfinite(close_hip_flexion) else "Close Hip Flexion: NA",
        f"Far Hip Flexion: {far_hip_flexion:.1f} deg" if np.isfinite(far_hip_flexion) else "Far Hip Flexion: NA",
        f"Far Knee Extension: {far_knee_extension:.1f} deg" if np.isfinite(far_knee_extension) else "Far Knee Extension: NA",
        f"Jurdan Angle: {jurdan_angle:.1f} deg" if np.isfinite(jurdan_angle) else "Jurdan Angle: NA",
        f"HipCheck Angle: {hipcheck_angle:.1f} deg" if np.isfinite(hipcheck_angle) else "HipCheck Angle: NA",
    ]
    x0, y0 = 12, 22
    for i, line in enumerate(panel):
        cv2.putText(draw, line, (x0, y0 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 4)  # shadow
        cv2.putText(draw, line, (x0, y0 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    metrics = {
        "close_side": close_side,
        "close_hip_flexion_deg": close_hip_flexion,
        "far_hip_flexion_deg": far_hip_flexion,
        "far_knee_extension_deg": far_knee_extension,
        "jurdan_angle_deg": jurdan_angle,
        "hipcheck_angle_deg": hipcheck_angle,
    }
    return draw, metrics

def detect_pose_and_optionally_adjust(file_obj, image_key: str):
    """
    Full pipeline for a single image: detect, show adjust UI, apply, annotate, compute metrics.
    Returns (annotated_image_bgr, metrics_dict) or (None, None) if detection failed.
    """
    if file_obj is None:
        return None, None

    # Prepare images for MediaPipe and OpenCV
    mp_img, image_bgr = mp_image_from_filelike(file_obj)
    model = load_model()
    results = model.detect(mp_img)

    if not results.pose_landmarks:
        st.error(f"{image_key}: No pose detected. Try a clearer side-view photo.")
        return None, None

    height, width = image_bgr.shape[:2]
    # MP returns a list of landmarks for the first person found
    landmarks = results.pose_landmarks[0]

    # Adjustment UI
    current_adjustments = create_adjustment_interface(image_bgr, landmarks, width, height, image_key)

    # Apply adjustments to landmarks (if any)
    adjusted_landmarks = apply_adjustments_to_landmarks(landmarks, current_adjustments)

    # Recompute metrics and draw annotated output
    annotated_bgr, metrics = annotate_and_compute(image_bgr, adjusted_landmarks, current_adjustments)
    return annotated_bgr, metrics

# =========================
# App UI
# =========================
st.markdown("Upload two side-view photos to compare. Clear lighting and full lower-limb visibility improves results.")

c1, c2 = st.columns(2)
with c1:
    file_a = st.file_uploader("📷 Image A", type=["jpg", "jpeg", "png"], key="upload_A")
with c2:
    file_b = st.file_uploader("📷 Image B", type=["jpg", "jpeg", "png"], key="upload_B")

# Process images
annot_a, metrics_a = detect_pose_and_optionally_adjust(file_a, "A")
annot_b, metrics_b = detect_pose_and_optionally_adjust(file_b, "B")

# Results
if annot_a is not None or annot_b is not None:
    st.markdown("### Results")
    img_cols = st.columns(2)
    if annot_a is not None:
        with img_cols[0]:
            st.image(cv2.cvtColor(annot_a, cv2.COLOR_BGR2RGB), caption="A • Annotated", use_container_width=True)
    if annot_b is not None:
        with img_cols[1]:
            st.image(cv2.cvtColor(annot_b, cv2.COLOR_BGR2RGB), caption="B • Annotated", use_container_width=True)

    # Metrics table
    rows = []
    if metrics_a is not None:
        rows.append({"Image": "A", **metrics_a})
    if metrics_b is not None:
        rows.append({"Image": "B", **metrics_b})

    if rows:
        df = pd.DataFrame(rows)
        # Order columns nicely
        cols = ["Image", "close_side", "close_hip_flexion_deg", "far_hip_flexion_deg",
                "far_knee_extension_deg", "jurdan_angle_deg", "hipcheck_angle_deg"]
        df = df[cols]
        st.dataframe(df, use_container_width=True)

        # Simple delta if both present
        if metrics_a is not None and metrics_b is not None:
            st.markdown("#### A vs B (B − A)")
            def safe_delta(b, a):
                if b is None or a is None or (not np.isfinite(b)) or (not np.isfinite(a)):
                    return np.nan
                return float(b - a)

            delta = {
                "Δ close_hip_flexion_deg": safe_delta(metrics_b["close_hip_flexion_deg"], metrics_a["close_hip_flexion_deg"]),
                "Δ far_hip_flexion_deg": safe_delta(metrics_b["far_hip_flexion_deg"], metrics_a["far_hip_flexion_deg"]),
                "Δ far_knee_extension_deg": safe_delta(metrics_b["far_knee_extension_deg"], metrics_a["far_knee_extension_deg"]),
                "Δ jurdan_angle_deg": safe_delta(metrics_b["jurdan_angle_deg"], metrics_a["jurdan_angle_deg"]),
                "Δ hipcheck_angle_deg": safe_delta(metrics_b["hipcheck_angle_deg"], metrics_a["hipcheck_angle_deg"]),
            }
            delta_df = pd.DataFrame([delta])
            st.dataframe(delta_df, use_container_width=True)

# Footer / Notes
st.caption(
    "Notes: "
    "Angles are computed from 2D landmarks; camera angle, occlusion, and clothing can affect accuracy. "
    "Manual joint correction overrides only the selected points (normalized to image size)."
)
