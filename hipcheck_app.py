import os
import sys
import base64
import urllib.request
from io import BytesIO
import platform
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Enable HEIC/HEIF reading seamlessly through Pillow
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass  # If the wheel isn't available locally yet, pip install will provide it

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

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Canvas (canonical package)
from streamlit_drawable_canvas import st_canvas

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

def pil_to_data_url(pil_img: Image.Image) -> str:
    """Convert PIL image to base64 PNG data URL for embedding into Fabric JSON."""
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def build_initial_drawing_with_embedded_bg(pil_disp: Image.Image, joint_px: Dict[str, Tuple[int, int]],
                                           point_radius: int = 8) -> Dict:
    """Embed the image as a Fabric 'image' object (locked) as the first object, then draggable joint circles."""
    w, h = pil_disp.size
    data_url = pil_to_data_url(pil_disp)

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
            left = obj.get("left")
            top  = obj.get("top")
            if left is not None and top is not None:
                out[obj["name"]] = (int(float(left)), int(float(top)))
    return out

def joints_px_from_landmarks(landmarks, w: int, h: int):
    d = {}
    for name, idx in JOINTS.items():
        lm = landmarks[idx]
        # Clamp to bounds for occasional tiny off-image values
        x = int(np.clip(lm.x * w, 0, w - 1))
        y = int(np.clip(lm.y * h, 0, h - 1))
        d[name] = (x, y)
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

# ---------------- front-end photo processing ----------------
def auto_enhance_pil(image: Image.Image) -> Image.Image:
    """
    Optional: mild CLAHE contrast + light denoise to help keypoints pop.
    Works on RGB, returns RGB.
    """
    try:
        import cv2
        arr = np.array(image)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        # CLAHE on L channel (LAB)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab2 = cv2.merge([cl, a, b])
        bgr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        # mild denoise
        bgr2 = cv2.fastNlMeansDenoisingColored(bgr2, None, h=3, hColor=3,
                                               templateWindowSize=7, searchWindowSize=21)
        rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb2)
    except Exception:
        # If OpenCV not available or anything fails, return original
        return image

def load_and_preprocess(uploaded_file, enhance: bool, rotation_state_key: str) -> Image.Image:
    """
    Read an uploaded image (JPG/PNG/HEIC), fix EXIF orientation, optionally enhance, and apply manual rotation.
    """
    # Open with Pillow (HEIC is supported if pillow-heif opener is registered)
    img = Image.open(uploaded_file)
    # EXIF orientation fix so portrait stays upright
    img = ImageOps.exif_transpose(img).convert("RGB")

    # Optional enhancement
    if enhance:
        img = auto_enhance_pil(img)

    # Manual rotation controls
    with st.columns([1, 1, 1], vertical_alignment="center") as cols:
        rot_left  = cols[0].button("⟲ Rotate −90°", key=rotation_state_key + "_l")
        rot_180   = cols[1].button("⤾ Rotate 180°", key=rotation_state_key + "_180")
        rot_right = cols[2].button("⟳ Rotate +90°", key=rotation_state_key + "_r")
    if rot_left:
        img = img.rotate(90, expand=True)
    if rot_right:
        img = img.rotate(-90, expand=True)
    if rot_180:
        img = img.rotate(180, expand=True)

    return img

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

def process_one_image(uploaded_file, image_key_hint: str, enhance_toggle_key: str):
    """
    Load (with HEIC support & EXIF orientation), optional auto-enhance,
    detect pose, build FULL‑RES canvas (no crop), and return (annotated, metrics, close_side).
    """
    if uploaded_file is None:
        return None, None, None

    file_id = f"{uploaded_file.name}-{getattr(uploaded_file, 'size', 'NA')}"
    rotation_state_key = f"rot_{image_key_hint}_{file_id}"

    # --- front-end processing ---
    enhance = st.toggle("Auto‑enhance (contrast + light denoise)", key=enhance_toggle_key, value=True)
    pil_full = load_and_preprocess(uploaded_file, enhance, rotation_state_key)

    # Preview (Streamlit scales for display only)
    st.image(pil_full, caption=f"{image_key_hint} preview (processed)", use_column_width=True)

    # Detect pose on processed full‑res image
    model = load_model()
    results = model.detect(mp_image_from_pil(pil_full))
    if not results.pose_landmarks:
        st.error(f"{image_key_hint}: No pose detected. Try clearer side-view, better lighting, or rotate.")
        return None, None, None
    landmarks = results.pose_landmarks[0]

    # Determine which side is closer using z (robust)
    def safe_mean_z(landmarks, idxs):
        vals = [getattr(landmarks[i], "z", None) for i in idxs]
        vals = [v for v in vals if v is not None]
        return float(np.nanmean(vals)) if vals else np.nan

    left_avg_z  = safe_mean_z(landmarks, LEFT_LMKS)
    right_avg_z = safe_mean_z(landmarks, RIGHT_LMKS)
    if np.isnan(left_avg_z) or np.isnan(right_avg_z):
        close_side_hint = "left"
    else:
        # Smaller (often negative) z is closer to camera
        close_side_hint = "left" if left_avg_z < right_avg_z else "right"

    # FULL‑RES canvas (no resizing/cropping)
    pil_disp = pil_full
    w_disp, h_disp = pil_disp.size

    joint_px0 = joints_px_from_landmarks(landmarks, w_disp, h_disp)

    st.markdown(f"### {image_key_hint} • Drag the joint dots (live-updating angles)")

    init_key = f"init_drawing_{image_key_hint}_{file_id}"
    if init_key not in st.session_state:
        st.session_state[init_key] = build_initial_drawing_with_embedded_bg(
            pil_disp, joint_px0, point_radius=8
        )

    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=0,
        stroke_color="rgba(0,0,0,0)",
        background_color="rgba(0,0,0,0)",
        background_image=None,          # We embed image as first Fabric object
        update_streamlit=True,
        height=h_disp,
        width=w_disp,
        drawing_mode="transform",
        initial_drawing=st.session_state[init_key],
        display_toolbar=True,
        point_display_radius=0,
        key=f"canvas_{image_key_hint}_{file_id}",
    )

    json_data = canvas_result.json_data or st.session_state[init_key]
    joint_px = extract_joint_px_from_canvas(json_data, joint_px0)

    metrics = compute_metrics_from_joint_px(joint_px, w_disp, h_disp, close_side_hint=close_side_hint)

    # Live values
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

    annot = annotate_with_joints(pil_disp, joint_px, metrics)

    if canvas_result.json_data:
        st.session_state[init_key] = canvas_result.json_data

    return annot, metrics, metrics.get("close_side", close_side_hint)

# ---------------- UI ----------------
st.markdown(
    "Upload up to **two** side‑view photos for the **same subject** lying on a table. "
    "The app auto‑orients (EXIF), can auto‑enhance visibility, and will auto‑assign each to "
    "**Left closer** or **Right closer** based on camera proximity."
)

files = st.file_uploader(
    "📷 Upload up to 2 images (JPG/PNG/HEIC)",
    type=["jpg", "jpeg", "png", "heic", "HEIC", "heif", "HEIF"],
    accept_multiple_files=True
)

# Take first two if more provided
if files and len(files) > 2:
    st.warning("You uploaded more than two images; using the first two.")
    files = files[:2]

left_pack  = None
right_pack = None

if files:
    for idx, f in enumerate(files, start=1):
        with st.container(border=True):
            annot, metrics, side = process_one_image(
                f,
                image_key_hint=f"Image {idx}",
                enhance_toggle_key=f"enh_{idx}"
            )
            if annot is None:
                continue
            if side == "left" and left_pack is None:
                left_pack = (annot, metrics)
            elif side == "right" and right_pack is None:
                right_pack = (annot, metrics)
            else:
                st.info(
                    f"This image also appears to be **{side} closer**. "
                    f"Capture the opposite side for a full comparison if needed."
                )

# --------- Results section ---------
if left_pack or right_pack:
    st.markdown("## Results (annotated overlays)")
    cols = st.columns(2)
    if left_pack:
        with cols[0]:
            st.subheader("Left closer")
            st.image(left_pack[0], use_column_width=True)
    if right_pack:
        with cols[1]:
            st.subheader("Right closer")
            st.image(right_pack[0], use_column_width=True)

    rows = []
    if left_pack:  rows.append({"Image": "Left closer",  **left_pack[1]})
    if right_pack: rows.append({"Image": "Right closer", **right_pack[1]})

    if rows:
        df = pd.DataFrame(rows)[
            ["Image","close_side","close_hip_flexion_deg","far_hip_flexion_deg",
             "far_knee_extension_deg","jurdan_angle_deg","hipcheck_angle_deg"]
        ]
        st.dataframe(df, use_container_width=True)

        if left_pack and right_pack:
            st.markdown("### Right − Left (Δ)")
            def d(b, a):
                if a is None or b is None:
                    return np.nan
                if not (np.isfinite(a) and np.isfinite(b)):
                    return np.nan
                return float(b - a)
            delta_df = pd.DataFrame([{
                "Δ close_hip_flexion_deg":  d(right_pack[1]["close_hip_flexion_deg"],  left_pack[1]["close_hip_flexion_deg"]),
                "Δ far_hip_flexion_deg":    d(right_pack[1]["far_hip_flexion_deg"],   left_pack[1]["far_hip_flexion_deg"]),
                "Δ far_knee_extension_deg": d(right_pack[1]["far_knee_extension_deg"], left_pack[1]["far_knee_extension_deg"]),
                "Δ jurdan_angle_deg":       d(right_pack[1]["jurdan_angle_deg"],       left_pack[1]["jurdan_angle_deg"]),
                "Δ hipcheck_angle_deg":     d(right_pack[1]["hipcheck_angle_deg"],     left_pack[1]["hipcheck_angle_deg"]),
            }])
            st.dataframe(delta_df, use_container_width=True)

st.caption(
    "Assumes subject is lying on a table. Images are orientation‑corrected using EXIF; "
    "optional auto‑enhance improves contrast/visibility. "
    "Drag the points to correct them; angles update live."
)
