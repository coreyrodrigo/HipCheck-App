import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
import tempfile
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request

st.set_page_config(page_title="HipCheck (Canvas Editor)", layout="centered")

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

def process_image(image_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(image_file.read())
    image = mp.Image.create_from_file(temp_file.name)
    model = load_model()
    results = model.detect(image)

    image_bgr = cv2.imread(temp_file.name)
    height, width, _ = image_bgr.shape

    if not results.pose_landmarks:
        st.warning("Pose not detected. You can manually mark joints below.")
        return None, image_bgr

    landmarks = results.pose_landmarks[0]
    joints = {}

    for i, name in zip([11, 23, 25, 27], ["shoulder", "hip", "knee", "ankle"]):
        lm = landmarks[i]
        joints[name] = (int(lm.x * width), int(lm.y * height))

    return joints, image_bgr

def calculate_angles(joints):
    def angle(a, b, c):
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    hip_angle = angle(joints['shoulder'], joints['hip'], joints['knee'])
    knee_angle = angle(joints['hip'], joints['knee'], joints['ankle'])
    hip_flexion = 180 - hip_angle
    knee_extension = knee_angle - 90
    jurdan_angle = hip_flexion + knee_extension
    return hip_flexion, knee_extension, jurdan_angle

st.title("HipCheck: Canvas Pose Editor")
username = st.text_input("Enter your name:")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file and username:
    joints, image_bgr = process_image(uploaded_file)
    img_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    st.markdown("### Mark joints manually if needed")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 255, 0.6)",
        stroke_width=5,
        background_image=img_pil,
        update_streamlit=True,
        height=img_pil.height,
        width=img_pil.width,
        drawing_mode="circle",
        key="canvas",
    )

    if canvas_result.json_data and 'objects' in canvas_result.json_data:
        points = [(int(obj['left']), int(obj['top'])) for obj in canvas_result.json_data['objects']]
        if len(points) >= 4:
            joints = dict(zip(["shoulder", "hip", "knee", "ankle"], points))
            flex, ext, jurdan = calculate_angles(joints)
            st.markdown("### Results")
            st.write(f"Hip Flexion: {flex:.1f}°")
            st.write(f"Knee Extension: {ext:.1f}°")
            st.write(f"**Jurdan Angle: {jurdan:.1f}°**")
        else:
            st.info("Mark at least 4 points: shoulder, hip, knee, and ankle.")
    elif joints:
        flex, ext, jurdan = calculate_angles(joints)
        st.markdown("### Auto-Detected Results")
        st.write(f"Hip Flexion: {flex:.1f}°")
        st.write(f"Knee Extension: {ext:.1f}°")
        st.write(f"**Jurdan Angle: {jurdan:.1f}°**")
