###############################################################
# FULL STREAMLIT POSE COMPARISON APP
# HEIC SUPPORT • FULL-RES CANVAS • AUTO-ORIENT • AUTO-ENHANCE
# LEFT-CLOSER / RIGHT-CLOSER ASSIGNMENT • STREAMLIT CLOUD SAFE
###############################################################

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

# Enable HEIC/HEIF support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

import streamlit as st
st.set_page_config(page_title="Pose Comparison", layout="centered")

# --- sanity check for OpenCV ---
try:
    import cv2
    st.info(f"OpenCV OK • cv2={cv2.__version__} • Python={sys.version.split()[0]}")
except Exception as e:
    st.error("OpenCV failed to import. Use opencv-python-headless.\n" + str(e))

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Canvas
from streamlit_drawable_canvas import st_canvas

APP_TITLE = "Pose Comparison"
st.title(APP_TITLE)

###############################################################
# MODEL DOWNLOAD — SAFE FOR STREAMLIT CLOUD
###############################################################

MODEL_DIR = "/tmp"
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_full.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    with st.status("Downloading MediaPipe Pose model…"):
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

###############################################################
# JOINTS
###############################################################

JOINTS = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_foot": 31, "right_foot": 32,
}
LEFT_LMKS  = [11, 23, 25, 27, 31]
RIGHT_LMKS = [12, 24, 26, 28, 32]

###############################################################
# HELPER FUNCTIONS
###############################################################

def mp_image_from_pil(pil_img: Image.Image) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))


def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return np.nan
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1, 1)
    return float(np.degrees(np.arccos(cosang)))


def draw_line(draw, p1, p2, width=6, fill=(255,255,0)):
    draw.line([p1, p2], width=width, fill=fill)


def draw_circle(draw, center, radius, fill, outline=(255,255,255), outline_width=2):
    x, y = center
    bbox = [x-radius, y-radius, x+radius, y+radius]
    draw.ellipse(bbox, fill=fill, outline=outline, width=outline_width)


def put_text(draw, xy, text, font=None, fill=(255,255,255)):
    if font is None:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 18)
        except:
            font = ImageFont.load_default()
    x, y = xy
    draw.text((x+1,y+1), text, font=font, fill=(0,0,0))
    draw.text((x,y), text, font=font, fill=fill)


def pil_to_data_url(img):
    buff = BytesIO()
    img.save(buff, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buff.getvalue()).decode()


def build_initial_canvas_json(pil_img, joint_px):
    w, h = pil_img.size
    bg = {
        "type":"image", "version":"5.2.4",
        "left":0, "top":0,
        "width":w, "height":h,
        "scaleX":1, "scaleY":1,
        "src": pil_to_data_url(pil_img),
        "selectable":False, "evented":False,
        "hasControls":False, "hasBorders":False,
        "lockMovementX":True,"lockMovementY":True,
        "lockScalingX":True,"lockScalingY":True,
        "lockRotation":True,
        "originX":"left","originY":"top"
    }
    circles = []
    for name,(x,y) in joint_px.items():
        circles.append({
            "type":"circle", "name":name,
            "left":float(x), "top":float(y),
            "radius":8,
            "fill":"rgba(0,200,0,0.85)",
            "stroke":"white",
            "strokeWidth":2,
            "selectable":True, "evented":True,
            "lockScalingX":True, "lockScalingY":True,
            "lockRotation":True,
            "originX":"center", "originY":"center"
        })
    return {"version":"5.2.4", "objects":[bg] + circles}


def extract_joint_px(json_data, fallback):
    if not json_data or "objects" not in json_data:
        return fallback
    out = dict(fallback)
    for obj in json_data["objects"]:
        if obj.get("type")=="circle" and obj.get("name") in out:
            left = obj.get("left")
            top  = obj.get("top")
            if left is not None and top is not None:
                out[obj["name"]] = (int(left), int(top))
    return out


def joints_px_from_landmarks(lmks, w, h):
    d = {}
    for name, idx in JOINTS.items():
        lm = lmks[idx]
        x = int(np.clip(lm.x * w, 0, w-1))
        y = int(np.clip(lm.y * h, 0, h-1))
        d[name] = (x,y)
    return d


def compute_metrics(joint_px, w, h, close_side):
    def P(name): return np.array([joint_px[name][0]/w, joint_px[name][1]/h])

    if close_side == "left":
        ch = calc_angle(P("left_shoulder"),P("left_hip"),P("left_knee"))
        fh = calc_angle(P("right_shoulder"),P("right_hip"),P("right_knee"))
        fk = calc_angle(P("right_hip"),P("right_knee"),P("right_ankle"))
    else:
        ch = calc_angle(P("right_shoulder"),P("right_hip"),P("right_knee"))
        fh = calc_angle(P("left_shoulder"),P("left_hip"),P("left_knee"))
        fk = calc_angle(P("left_hip"),P("left_knee"),P("left_ankle"))

    chf = 180-ch if np.isfinite(ch) else np.nan
    fhf = 180-fh if np.isfinite(fh) else np.nan
    fke = (fk-90) if np.isfinite(fk) else np.nan
    jurdan = (chf + fke) if np.isfinite(chf) and np.isfinite(fke) else np.nan
    hipcheck = (jurdan - (90 - fhf)) if np.isfinite(jurdan) and np.isfinite(fhf) else np.nan

    return {
        "close_side": close_side,
        "close_hip_flexion_deg": chf,
        "far_hip_flexion_deg": fhf,
        "far_knee_extension_deg": fke,
        "jurdan_angle_deg": jurdan,
        "hipcheck_angle_deg": hipcheck,
    }


def annotate(pil_img, joint_px, metrics):
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    for a,b in [
        ("left_shoulder","left_hip"),("left_hip","left_knee"),
        ("left_knee","left_ankle"),("left_ankle","left_foot"),
        ("right_shoulder","right_hip"),("right_hip","right_knee"),
        ("right_knee","right_ankle"),("right_ankle","right_foot")
    ]:
        draw_line(draw, joint_px[a], joint_px[b])
    for name,pos in joint_px.items():
        draw_circle(draw,pos,8,(220,0,0))
    try: font = ImageFont.truetype("DejaVuSans.ttf",18)
    except: font = ImageFont.load_default()
    panel = [
        f"Close side: {metrics['close_side']}",
        f"Close Hip Flexion: {metrics['close_hip_flexion_deg']:.1f}°",
        f"Far Hip Flexion: {metrics['far_hip_flexion_deg']:.1f}°",
        f"Far Knee Extension: {metrics['far_knee_extension_deg']:.1f}°",
        f"Jurdan: {metrics['jurdan_angle_deg']:.1f}°",
        f"HipCheck: {metrics['hipcheck_angle_deg']:.1f}°",
    ]
    y=20
    for line in panel:
        put_text(draw,(20,y),line,font)
        y+=22
    return img

###############################################################
# FRONT-END IMAGE PROCESSING: EXIF ORIENT + OPTIONAL ENHANCE
###############################################################

def auto_enhance(img):
    try:
        import cv2
        arr = np.array(img)
        bgr = cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB)
        L,a,b = cv2.split(lab)
        clahe = cv2.createCLAHE(2.0,(8,8))
        L2 = clahe.apply(L)
        lab2 = cv2.merge([L2,a,b])
        bgr2 = cv2.cvtColor(lab2,cv2.COLOR_LAB2BGR)
        bgr2 = cv2.fastNlMeansDenoisingColored(bgr2,None,3,3,7,21)
        rgb = cv2.cvtColor(bgr2,cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    except:
        return img


def load_and_preprocess(file, enhance, rotation_key):
    img = Image.open(file)
    img = ImageOps.exif_transpose(img).convert("RGB")   # fix upright orientation

    if enhance:
        img = auto_enhance(img)

    # ROTATION BUTTONS — FIXED FOR STREAMLIT CLOUD
    cols = st.columns([1,1,1])
    if cols[0].button("⟲ -90°", key=rotation_key+"_l"):
        img = img.rotate(90, expand=True)
    if cols[1].button("⤾ 180°", key=rotation_key+"_180"):
        img = img.rotate(180, expand=True)
    if cols[2].button("⟳ +90°", key=rotation_key+"_r"):
        img = img.rotate(-90, expand=True)

    return img

###############################################################
# MAIN IMAGE PROCESSOR
###############################################################

def process_one(file, label, enhance_key):
    if file is None:
        return None,None,None

    rotation_key = f"rot_{label}_{file.name}"

    # FRONT-END PROCESSING
    enhance = st.checkbox(f"Auto-enhance {label}", key=enhance_key, value=True)
    img = load_and_preprocess(file, enhance, rotation_key)

    st.image(img, caption=f"{label} (processed)", use_column_width=True)

    # POSE DETECTION
    model = load_model()
    results = model.detect(mp_image_from_pil(img))
    if not results.pose_landmarks:
        st.error(f"{label}: Pose not detected. Try rotating or enhancing.")
        return None,None,None
    lmks = results.pose_landmarks[0]

    # DETERMINE WHICH SIDE IS CLOSER
    def mean_z(indices):
        vals=[getattr(lmks[i],"z",None) for i in indices]
        vals=[v for v in vals if v is not None]
        return float(np.nanmean(vals)) if vals else np.nan

    lz = mean_z(LEFT_LMKS)
    rz = mean_z(RIGHT_LMKS)

    if np.isnan(lz) or np.isnan(rz):
        side="left"
    else:
        side="left" if lz < rz else "right"

    # FULL-RES CANVAS
    w,h = img.size
    joints0 = joints_px_from_landmarks(lmks,w,h)

    st.markdown(f"### {label}: Drag Joints")

    key=f"init_{label}_{file.name}"
    if key not in st.session_state:
        st.session_state[key] = build_initial_canvas_json(img, joints0)

    canvas = st_canvas(
        fill_color="rgba(0,0,0,0)", stroke_width=0,
        background_color="rgba(0,0,0,0)",
        update_streamlit=True,
        height=h, width=w,
        drawing_mode="transform",
        initial_drawing=st.session_state[key],
        display_toolbar=True,
        key=f"canvas_{label}_{file.name}"
    )

    json_data = canvas.json_data or st.session_state[key]
    joints = extract_joint_px(json_data, joints0)
    metrics = compute_metrics(joints,w,h,side)

    # Live numbers
    def fmt(x): return f"{x:.1f}°" if x is not None and np.isfinite(x) else "NA"
    st.write(f"**Close Hip Flexion:** {fmt(metrics['close_hip_flexion_deg'])}")
    st.write(f"**Far Hip Flexion:** {fmt(metrics['far_hip_flexion_deg'])}")
    st.write(f"**Far Knee Extension:** {fmt(metrics['far_knee_extension_deg'])}")
    st.write(f"**Jurdan:** {fmt(metrics['jurdan_angle_deg'])}")
    st.write(f"**HipCheck:** {fmt(metrics['hipcheck_angle_deg'])}")

    if canvas.json_data:
        st.session_state[key]=canvas.json_data

    annot = annotate(img,joints,metrics)
    return annot, metrics, side

###############################################################
# UI
###############################################################

st.markdown("""
Upload **two** images (JPG/PNG/HEIC).  
The subject should be **lying on a table**.  
The app auto‑orients, auto‑enhances, and auto‑assigns each image  
to **Left Closer** or **Right Closer** based on camera depth.
""")

files = st.file_uploader(
    "Upload up to 2 images",
    type=["jpg","jpeg","png","heic","HEIC","heif","HEIF"],
    accept_multiple_files=True
)

if files and len(files)>2:
    st.warning("Using first two images.")
    files = files[:2]

left_pack=None
right_pack=None

if files:
    for idx,f in enumerate(files,1):
        with st.container(border=True):
            annot,metrics,side = process_one(f,f"Image {idx}",f"enh_{idx}")
            if annot:
                if side=="left" and left_pack is None:
                    left_pack=(annot,metrics)
                elif side=="right" and right_pack is None:
                    right_pack=(annot,metrics)
                else:
                    st.info(f"Image {idx} also appears to be {side}-closer.")

###############################################################
# RESULTS
###############################################################

if left_pack or right_pack:
    st.header("Results")

    cols=st.columns(2)
    if left_pack:
        with cols[0]:
            st.subheader("Left Closer")
            st.image(left_pack[0], use_column_width=True)
    if right_pack:
        with cols[1]:
            st.subheader("Right Closer")
            st.image(right_pack[0], use_column_width=True)

    rows=[]
    if left_pack:  rows.append({"Image":"Left Closer", **left_pack[1]})
    if right_pack: rows.append({"Image":"Right Closer",**right_pack[1]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if left_pack and right_pack:
        st.subheader("Right − Left Δ")
        def d(b,a):
            if not (np.isfinite(a) and np.isfinite(b)): return np.nan
            return float(b-a)
        L=left_pack[1]; R=right_pack[1]
        delta_df=pd.DataFrame([{
            "Δ close_hip_flexion_deg":d(R["close_hip_flexion_deg"],L["close_hip_flexion_deg"]),
            "Δ far_hip_flexion_deg":d(R["far_hip_flexion_deg"],L["far_hip_flexion_deg"]),
            "Δ far_knee_extension_deg":d(R["far_knee_extension_deg"],L["far_knee_extension_deg"]),
            "Δ jurdan_angle_deg":d(R["jurdan_angle_deg"],L["jurdan_angle_deg"]),
            "Δ hipcheck_angle_deg":d(R["hipcheck_angle_deg"],L["hipcheck_angle_deg"]),
        }])
        st.dataframe(delta_df,use_container_width=True)

st.caption("All processing is done front‑end for accuracy and ease of use.")
