import os
import sys
import base64
import urllib.request
from io import BytesIO
import platform
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# HEIC support (no orientation correction)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

import streamlit as st
st.set_page_config(page_title="Pose Comparison", layout="centered")

# sanity check for OpenCV
try:
    import cv2
except:
    pass

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from streamlit_drawable_canvas import st_canvas

############################################################
# TITLE
############################################################
st.title("Pose Comparison")

############################################################
# MODEL SETUP
############################################################

MODEL_DIR = "/tmp"
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_full.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

os.makedirs(MODEL_DIR, exist_ok=True)
if not os.path.exists(MODEL_PATH):
    with st.status("Downloading pose model…"):
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

############################################################
# JOINTS
############################################################

JOINTS = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_foot": 31, "right_foot": 32,
}

LEFT_LMKS  = [11,23,25,27,31]
RIGHT_LMKS = [12,24,26,28,32]

############################################################
# HELPERS
############################################################

def mp_image_from_pil(img: Image.Image) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img))


def calc_angle(a,b,c):
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return np.nan
    cosang = np.dot(ba,bc)/denom
    cosang = np.clip(cosang,-1,1)
    return float(np.degrees(np.arccos(cosang)))


def draw_line(draw,p1,p2,width=6,fill=(255,255,0)):
    draw.line([p1,p2],width=width,fill=fill)


def draw_circle(draw,center,radius,fill=(220,0,0),
                outline=(255,255,255),outline_width=2):
    x,y=center
    bbox=[x-radius,y-radius,x+radius,y+radius]
    draw.ellipse(bbox,fill=fill,outline=outline,width=outline_width)


def put_text(draw,xy,text,font=None,fill=(255,255,255)):
    if font is None:
        try: font=ImageFont.truetype("DejaVuSans.ttf",18)
        except: font=ImageFont.load_default()
    x,y=xy
    draw.text((x+1,y+1),text,font=font,fill=(0,0,0))
    draw.text((x,y),text,font=font,fill=fill)


def pil_to_data_url(img):
    buff=BytesIO()
    img.save(buff,format="PNG")
    return "data:image/png;base64," + base64.b64encode(buff.getvalue()).decode()


def build_canvas_json(img,joints):
    w,h=img.size
    bg={
        "type":"image",
        "version":"5.2.4",
        "left":0,"top":0,
        "width":w,"height":h,
        "scaleX":1,"scaleY":1,
        "src":pil_to_data_url(img),
        "selectable":False,"evented":False,
        "hasControls":False,"hasBorders":False,
        "lockMovementX":True,"lockMovementY":True,
        "lockScalingX":True,"lockScalingY":True,
        "lockRotation":True,
        "originX":"left","originY":"top"
    }
    circles=[]
    for name,(x,y) in joints.items():
        circles.append({
            "type":"circle","name":name,
            "left":float(x),"top":float(y),
            "radius":8,
            "fill":"rgba(0,200,0,0.85)",
            "stroke":"white","strokeWidth":2,
            "selectable":True,"evented":True,
            "lockScalingX":True,"lockScalingY":True,
            "lockRotation":True,
            "originX":"center","originY":"center"
        })
    return {"version":"5.2.4","objects":[bg]+circles}


def extract_joints(json_data,fallback):
    if not json_data or "objects" not in json_data:
        return fallback
    out=dict(fallback)
    for obj in json_data["objects"]:
        if obj.get("type")=="circle":
            name=obj.get("name")
            if name in out:
                left=obj.get("left")
                top =obj.get("top")
                if left is not None and top is not None:
                    out[name]=(int(left),int(top))
    return out


def joints_from_landmarks(landmarks,w,h):
    out={}
    for name,idx in JOINTS.items():
        lm=landmarks[idx]
        x=int(np.clip(lm.x*w,0,w-1))
        y=int(np.clip(lm.y*h,0,h-1))
        out[name]=(x,y)
    return out


def compute_metrics(joints,w,h,side):
    def P(n): 
        x,y=joints[n]
        return np.array([x/w,y/h])
    if side=="left":
        ch=calc_angle(P("left_shoulder"),P("left_hip"),P("left_knee"))
        fh=calc_angle(P("right_shoulder"),P("right_hip"),P("right_knee"))
        fk=calc_angle(P("right_hip"),P("right_knee"),P("right_ankle"))
    else:
        ch=calc_angle(P("right_shoulder"),P("right_hip"),P("right_knee"))
        fh=calc_angle(P("left_shoulder"),P("left_hip"),P("left_knee"))
        fk=calc_angle(P("left_hip"),P("left_knee"),P("left_ankle"))

    chf=180-ch if np.isfinite(ch) else np.nan
    fhf=180-fh if np.isfinite(fh) else np.nan
    fke=fk-90 if np.isfinite(fk) else np.nan
    jurdan=(chf+fke) if np.isfinite(chf) and np.isfinite(fke) else np.nan
    hipchk=(jurdan-(90-fhf)) if np.isfinite(jurdan) and np.isfinite(fhf) else np.nan

    return {
        "close_side":side,
        "close_hip_flexion_deg":chf,
        "far_hip_flexion_deg":fhf,
        "far_knee_extension_deg":fke,
        "jurdan_angle_deg":jurdan,
        "hipcheck_angle_deg":hipchk
    }


def annotate(img,joints,metrics):
    out=img.copy()
    draw=ImageDraw.Draw(out)
    # skeleton
    for a,b in [
        ("left_shoulder","left_hip"),("left_hip","left_knee"),
        ("left_knee","left_ankle"),("left_ankle","left_foot"),
        ("right_shoulder","right_hip"),("right_hip","right_knee"),
        ("right_knee","right_ankle"),("right_ankle","right_foot")
    ]:
        draw_line(draw,joints[a],joints[b])
    # joints
    for pos in joints.values():
        draw_circle(draw,pos)
    # metrics text
    try: font=ImageFont.truetype("DejaVuSans.ttf",18)
    except: font=ImageFont.load_default()
    y=20
    for key,val in metrics.items():
        put_text(draw,(20,y),f"{key}: {val:.1f}" if isinstance(val,float) else f"{key}: {val}",font)
        y+=22
    return out

############################################################
# MAIN PER-IMAGE PROCESSING
############################################################

def process_image(file,label):
    if file is None: 
        return None,None,None

    # Load EXACTLY as provided — NO orientation correction, NO enhancement
    img = Image.open(file).convert("RGB")

    st.image(img, caption=f"{label} (original)", use_column_width=True)

    model=load_model()
    res=model.detect(mp_image_from_pil(img))
    if not res.pose_landmarks:
        st.error(f"{label}: No pose detected.")
        return None,None,None
    lmks=res.pose_landmarks[0]

    # determine left vs right closer
    def mean_z(indices):
        vals=[getattr(lmks[i],"z",None) for i in indices]
        vals=[v for v in vals if v is not None]
        return float(np.nanmean(vals)) if vals else np.nan

    lz=mean_z(LEFT_LMKS)
    rz=mean_z(RIGHT_LMKS)
    side="left" if (not np.isnan(lz) and not np.isnan(rz) and lz<rz) else "right"

    w,h=img.size
    joints0=joints_from_landmarks(lmks,w,h)

    st.markdown(f"### {label} – Drag joints")

    key=f"canvas_init_{label}_{file.name}"
    if key not in st.session_state:
        st.session_state[key]=build_canvas_json(img,joints0)

    canvas=st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=0,
        background_color="rgba(0,0,0,0)",
        update_streamlit=True,
        height=h,width=w,
        drawing_mode="transform",
        initial_drawing=st.session_state[key],
        display_toolbar=True,
        key=f"canvas_{label}_{file.name}"
    )

    json_data=canvas.json_data or st.session_state[key]
    joints=extract_joints(json_data,joints0)
    metrics=compute_metrics(joints,w,h,side)

    annot = annotate(img,joints,metrics)

    if canvas.json_data:
        st.session_state[key]=canvas.json_data

    return annot,metrics,side

############################################################
# UI
############################################################

st.markdown("Upload EXACT images, unmodified. The app will NOT rotate, NOT crop, NOT enhance.")

files = st.file_uploader(
    "Upload up to 2 images (JPG/PNG/HEIC)",
    type=["jpg","jpeg","png","heic","HEIC","heif","HEIF"],
    accept_multiple_files=True
)

if files and len(files)>2:
    st.warning("Using first two images.")
    files = files[:2]

left_pack=None
right_pack=None

if files:
    for i,f in enumerate(files,1):
        with st.container(border=True):
            annot,metrics,side = process_image(f,f"Image {i}")
            if annot is None:
                continue
            if side=="left" and left_pack is None:
                left_pack=(annot,metrics)
            elif side=="right" and right_pack is None:
                right_pack=(annot,metrics)
            else:
                st.info(f"Image {i} also appears to be {side}-closer.")

############################################################
# RESULTS
############################################################

if left_pack or right_pack:
    st.header("Results")

    cols=st.columns(2)
    if left_pack:
        with cols[0]:
            st.subheader("Left Closer")
            st.image(left_pack[0],use_column_width=True)
    if right_pack:
        with cols[1]:
            st.subheader("Right Closer")
            st.image(right_pack[0],use_column_width=True)

    rows=[]
    if left_pack: rows.append({"Image":"Left Closer",**left_pack[1]})
    if right_pack: rows.append({"Image":"Right Closer",**right_pack[1]})
    st.dataframe(pd.DataFrame(rows),use_container_width=True)

    if left_pack and right_pack:
        st.subheader("Right − Left Δ")
        def d(b,a):
            if not (np.isfinite(a) and np.isfinite(b)): return np.nan
            return float(b-a)
        L=left_pack[1]; R=right_pack[1]
        delta=pd.DataFrame([{
            "Δ close_hip_flexion":d(R["close_hip_flexion_deg"],L["close_hip_flexion_deg"]),
            "Δ far_hip_flexion":d(R["far_hip_flexion_deg"],L["far_hip_flexion_deg"]),
            "Δ far_knee_ext":d(R["far_knee_extension_deg"],L["far_knee_extension_deg"]),
            "Δ jurdan":d(R["jurdan_angle_deg"],L["jurdan_angle_deg"]),
            "Δ hipcheck":d(R["hipcheck_angle_deg"],L["hipcheck_angle_deg"])
        }])
        st.dataframe(delta,use_container_width=True)

st.caption("This version uses EXACT original images: no auto-rotation, no enhancement, no cropping.")
