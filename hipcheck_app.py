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
import base64
from streamlit_dragzone import dragzone

# === Download model if not present ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")
if not os.path.exists(MODEL_PATH):
    st.info("Downloading pose model...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

@st.cache_resource
def load_model():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=vision.RunningMode.IMAGE
    )
    return vision.PoseLandmarker.create_from_options(options)

def launch_drag_editor(image_bgr):
    canvas_w, canvas_h = 800, 600
    img_resized = cv2.resize(image_bgr, (canvas_w, canvas_h))
    _, buffer = cv2.imencode(".png", img_resized)
    img_b64 = base64.b64encode(buffer).decode()

    st.markdown("### Manual Joint Correction: Drag the points to the correct location")
    html = f"""
        <img id="bg-img" src="data:image/png;base64,{img_b64}" style="position:absolute; z-index:-1; width:{canvas_w}px; height:{canvas_h}px;">
    """

    drag_points = dragzone(
        raw_html=html,
        children=[
            {"x": 300, "y": 450, "child": "🟥 Shoulder"},
            {"x": 320, "y": 500, "child": "🟦 Hip"},
            {"x": 400, "y": 550, "child": "🟩 Knee"},
            {"x": 500, "y": 580, "child": "🟨 Ankle"},
        ],
        height=canvas_h,
        width=canvas_w,
    )

    pos_dict = {pt['child']: (pt['x'], pt['y']) for pt in drag_points}
    shoulder = np.array(pos_dict["🟥 Shoulder"])
    hip = np.array(pos_dict["🟦 Hip"])
    knee = np.array(pos_dict["🟩 Knee"])
    ankle = np.array(pos_dict["🟨 Ankle"])

    def calc_angle(a, b, c):
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    hip_angle = calc_angle(shoulder, hip, knee)
    knee_angle = calc_angle(hip, knee, ankle)

    hip_flexion = 180 - hip_angle
    knee_extension = knee_angle - 90
    jurdan_angle = hip_flexion + knee_extension

    st.markdown("### Updated Angles")
    st.write(f"Hip Flexion: {hip_flexion:.1f}")
    st.write(f"Knee Extension: {knee_extension:.1f}")
    st.write(f"**Jurdan Angle: {jurdan_angle:.1f}**")

    return hip_flexion, knee_extension, jurdan_angle

def process_image(image_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(image_file.read())
    image = mp.Image.create_from_file(temp_file.name)
    model = load_model()
    results = model.detect(image)

    image_bgr = cv2.imread(temp_file.name)

    if not results.pose_landmarks:
        st.warning("Pose not detected. Use manual drag mode below.")
        return None, None, None, image_bgr

    height, width, _ = image_bgr.shape
    landmarks = results.pose_landmarks[0]

    def to_px(idx):
        lm = landmarks[idx]
        return int(lm.x * width), int(lm.y * height)

    def get_xy(idx):
        return np.array([landmarks[idx].x, landmarks[idx].y])

    def calc_angle(a, b, c):
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    JOINTS = {
        "left_shoulder": 11, "right_shoulder": 12,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28,
    }

    LEFT_LMKS = [11, 23, 25, 27]
    RIGHT_LMKS = [12, 24, 26, 28]
    left_avg_z = np.mean([landmarks[i].z for i in LEFT_LMKS])
    right_avg_z = np.mean([landmarks[i].z for i in RIGHT_LMKS])
    close_side = 'left' if left_avg_z < right_avg_z else 'right'

    if close_side == "left":
        hip_angle = calc_angle(get_xy(11), get_xy(23), get_xy(25))
        knee_angle = calc_angle(get_xy(23), get_xy(25), get_xy(27))
    else:
        hip_angle = calc_angle(get_xy(12), get_xy(24), get_xy(26))
        knee_angle = calc_angle(get_xy(24), get_xy(26), get_xy(28))

    hip_flexion = 180 - hip_angle
    knee_extension = knee_angle - 90
    jurdan_angle = hip_flexion + knee_extension

    return hip_flexion, knee_extension, jurdan_angle, image_bgr

# === Streamlit App ===
st.title("HipCheck: Jurdan Angle Analysis")
username = st.text_input("Enter user name:")
uploaded_file = st.file_uploader("Upload a single image", type=["jpg", "jpeg", "png"])

if uploaded_file and username:
    use_manual = st.checkbox("Override with manual drag editing", value=False)
    flex, ext, jurdan, img = process_image(uploaded_file)

    if use_manual or flex is None:
        flex, ext, jurdan = launch_drag_editor(img)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Pose with Landmarks", use_container_width=True)

    st.markdown("### Results")
    st.write(f"Hip Flexion: {flex:.1f}")
    st.write(f"Knee Extension: {ext:.1f}")
    st.write(f"**Jurdan Angle: {jurdan:.1f}**")
