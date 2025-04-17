import streamlit as st
from streamlit_drawable_canvas import st_canvas
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
from fpdf import FPDF

st.set_page_config(page_title="Pose Comparison", layout="centered")

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
        output_segmentation_masks=True,
        running_mode=vision.RunningMode.IMAGE
    )
    return vision.PoseLandmarker.create_from_options(options)

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

def process_image(image_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(image_file.read())
    image = mp.Image.create_from_file(temp_file.name)
    model = load_model()
    results = model.detect(image)
    image_bgr = cv2.imread(temp_file.name)
    height, width, _ = image_bgr.shape

    if not results.pose_landmarks:
        return None, image_bgr, height, width

    landmarks = results.pose_landmarks[0]
    JOINTS = {"shoulder": 11, "hip": 23, "knee": 25, "ankle": 27}
    joints = {name: (int(landmarks[i].x * width), int(landmarks[i].y * height)) for name, i in JOINTS.items()}

    return joints, image_bgr, height, width

st.title("Check Hip Dissociation")
username = st.text_input("Enter user name:")
uploaded_files = st.file_uploader("Upload 1 or 2 Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files and username:
    all_data = []
    images = []
    for file in uploaded_files:
        joints, image_bgr, height, width = process_image(file)
        img_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        if joints is None:
            st.warning("Pose not detected. Please mark shoulder, hip, knee, and ankle.")
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 255, 0.6)",
                stroke_width=5,
                background_image=img_pil,
                update_streamlit=True,
                height=height,
                width=width,
                drawing_mode="circle",
                key=f"canvas_{file.name}",
            )
            if canvas_result.json_data and 'objects' in canvas_result.json_data:
                pts = [(int(obj['left']), int(obj['top'])) for obj in canvas_result.json_data['objects']]
                if len(pts) >= 4:
                    joints = dict(zip(["shoulder", "hip", "knee", "ankle"], pts))
                else:
                    st.error("Mark 4 points.")
                    continue
            else:
                continue

        flexion, extension, jurdan = calculate_angles(joints)
        images.append((img_pil, jurdan))
        all_data.append({
            "Image Name": file.name,
            "Hip Flexion": round(flexion, 1),
            "Knee Extension": round(extension, 1),
            "Jurdan Angle": round(jurdan, 1)
        })

    if len(all_data) == 1:
        st.image(images[0][0], caption=uploaded_files[0].name, use_container_width=True)
        st.markdown("### Jurdan Angle")
        st.write(f"Hip Flexion: {all_data[0]['Hip Flexion']}°")
        st.write(f"Knee Extension: {all_data[0]['Knee Extension']}°")
        st.write(f"Jurdan Angle: {all_data[0]['Jurdan Angle']}°")

    elif len(all_data) == 2:
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(images[0][0], caption=uploaded_files[0].name, use_container_width=True)
        with img_col2:
            st.image(images[1][0], caption=uploaded_files[1].name, use_container_width=True)

        st.markdown("### Jurdan Angles")
        st.write(f"{uploaded_files[0].name}: {all_data[0]['Jurdan Angle']}°")
        st.write(f"{uploaded_files[1].name}: {all_data[1]['Jurdan Angle']}°")

        diff = abs(all_data[0]['Jurdan Angle'] - all_data[1]['Jurdan Angle'])
        st.write(f"**Difference: {diff:.1f}{' ⚠️' if diff > 15 else ''}**")

        now = datetime.now()
        df = pd.DataFrame([{
            "username": username,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            **all_data[0],
            **{f"{k}_2": v for k, v in all_data[1].items()},
            "jurdan_diff": round(diff, 1)
        }])

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Comparison CSV", data=csv,
                           file_name=f"{username}_jurdan_comparison_{now.strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")

        pdf = FPDF(orientation='L', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font("Arial", "B", 24)
        pdf.cell(0, 10, f"{username} - Jurdan Angle Comparison", ln=True, align="C")

        w_half = 140
        for i, (img, _) in enumerate(images):
            path = os.path.join(tempfile.gettempdir(), f"img{i}.jpg")
            img.save(path)
            pdf.image(path, x=10 + i * 140, y=30, w=w_half)

        pdf.set_xy(10, 135)
        pdf.set_font("Arial", size=16)
        pdf.cell(w_half, 10, f"{uploaded_files[0].name}: {all_data[0]['Jurdan Angle']}°", align="C")

        pdf.set_xy(150, 135)
        pdf.cell(w_half, 10, f"{uploaded_files[1].name}: {all_data[1]['Jurdan Angle']}°", align="C")

        pdf.set_xy(10, 155)
        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 10, f"Difference: {diff:.1f}°", ln=True, align="C")

        pdf_bytes = pdf.output(dest='S').encode('latin1')
        st.download_button("Download PDF Report", data=pdf_bytes,
                           file_name=f"{username}_jurdan_report.pdf", mime="application/pdf")
