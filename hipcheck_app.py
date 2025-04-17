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

def process_image(image_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(image_file.read())
    image = mp.Image.create_from_file(temp_file.name)
    model = load_model()
    results = model.detect(image)

    if not results.pose_landmarks:
        return None, None, None, None, None

    image_bgr = cv2.imread(temp_file.name)
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
        "left_foot": 31, "right_foot": 32
    }

    LEFT_LMKS = [11, 23, 25, 27, 31]
    RIGHT_LMKS = [12, 24, 26, 28, 32]
    left_avg_z = np.mean([landmarks[i].z for i in LEFT_LMKS])
    right_avg_z = np.mean([landmarks[i].z for i in RIGHT_LMKS])
    close_side = 'left' if left_avg_z < right_avg_z else 'right'

    if close_side == "left":
        close_hip_angle = calc_angle(get_xy(11), get_xy(23), get_xy(25))
        far_hip_angle = (calc_angle(get_xy(12), get_xy(24), get_xy(26)) -180 )*-1
        far_knee_angle = calc_angle(get_xy(24), get_xy(26), get_xy(28))
        close_hip_px = to_px(JOINTS["left_hip"])
        far_hip_px = to_px(JOINTS["right_hip"])
        far_knee_px = to_px(JOINTS["right_knee"])
    else:
        close_hip_angle = calc_angle(get_xy(12), get_xy(24), get_xy(26))
        far_hip_angle = (calc_angle(get_xy(11), get_xy(23), get_xy(25))  - 180)*-1
        far_knee_angle = calc_angle(get_xy(23), get_xy(25), get_xy(27))
        close_hip_px = to_px(JOINTS["right_hip"])
        far_hip_px = to_px(JOINTS["left_hip"])
        far_knee_px = to_px(JOINTS["left_knee"])


    close_hip_flexion = 180 - close_hip_angle
    far_hip_flexion = 180 - far_hip_angle
    far_knee_extension = far_knee_angle - 90
    jurdan_angle = close_hip_flexion + far_knee_extension
    hipcheck_angle = jurdan_angle - far_hip_flexion

    # === Drawing (unchanged) ===
    def draw_joint_line(a, b):
        cv2.line(image_bgr, to_px(JOINTS[a]), to_px(JOINTS[b]), (0, 255, 255), 6)

    for pair in [("left_shoulder", "left_hip"), ("left_hip", "left_knee"),
                 ("left_knee", "left_ankle"), ("left_ankle", "left_foot"),
                 ("right_shoulder", "right_hip"), ("right_hip", "right_knee"),
                 ("right_knee", "right_ankle"), ("right_ankle", "right_foot")]:
        draw_joint_line(*pair)

    for idx in JOINTS.values():
        cv2.circle(image_bgr, to_px(idx), 10, (0, 0, 255), -1)

    def draw_label(text, pos):
        cv2.putText(image_bgr, text, (pos[0] + 10, pos[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)

    draw_label(f"{close_hip_flexion:.1f}", close_hip_px)
    draw_label(f"{far_knee_extension:.1f}", far_knee_px)
    draw_label(f"{far_hip_angle:.1f}", far_hip_px)  # ← Add this

    def draw_jurdan_label(img, text):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 5)[0]
        center_x = (img.shape[1] - text_size[0]) // 2
        cv2.rectangle(img, (center_x - 40, 60), (center_x + text_size[0] + 40, 160), (0, 0, 0), -1)
        cv2.putText(img, text, (center_x, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 5, cv2.LINE_AA)

    draw_jurdan_label(image_bgr, f"Jurdan: {jurdan_angle:.1f} | HipCheck: {hipcheck_angle:.1f}")
    return close_side, jurdan_angle, (close_hip_flexion, far_knee_extension), image_bgr, hipcheck_angle, far_hip_angle


st.title("Check Hip Dissociation")

username = st.text_input("Enter Patient Name:")
uploaded_files = st.file_uploader("Upload 1 or 2 Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files and username:
    if len(uploaded_files) == 1:
        file = uploaded_files[0]
        side, jurdan, (flexion, extension), img, hipcheck, far_hip_angle = process_image(file)

        if side is None:
            st.error("Pose not detected in the image.")
        else:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"{side.capitalize()} Closer", use_container_width=True)
            st.markdown("### Jurdan Angle")
            st.write(f"{side.capitalize()} Jurdan Angle: {jurdan:.1f}")
            st.write(f"Hip Flexion: {flexion:.1f}")
            st.write(f"Knee Extension: {extension:.1f}")
            st.write(f"HipCheck Angle: {hipcheck:.1f}")
            st.write(f"Far Hip Angle: {far_hip_angle:.1f}")


    elif len(uploaded_files) == 2:
        file1, file2 = uploaded_files
        side1, jurdan1, (flex1, ext1), img1, hipcheck1, far_hip_angle1 = process_image(file1)
        side2, jurdan2, (flex2, ext2), img2, hipcheck2, far_hip_angle2 = process_image(file2)

        if side1 is None or side2 is None:
            st.error("Pose not detected in one or both images.")
        else:
            angles = {side1: jurdan1, side2: jurdan2}
            left_angle = angles.get('left')
            right_angle = angles.get('right')

            if left_angle is not None and right_angle is not None:
                diff = abs(left_angle - right_angle)
                st.markdown("### Pose Images with Keypoints")
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.image(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), caption=f"{side1.capitalize()} Closer", use_container_width=True)
                with img_col2:
                    st.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), caption=f"{side2.capitalize()} Closer", use_container_width=True)

                st.markdown("### Jurdan Angles")
                st.write(f"Left Jurdan Angle: {left_angle:.1f}")
                st.write(f"Right Jurdan Angle: {right_angle:.1f}")
                st.write(f"**Difference: {diff:.1f}{' ⚠️' if diff > 15 else ''}**")
                st.write(
                    f"Left Far Hip Angle: {far_hip_angle1:.1f}" if side1 == "left" else f"Right Far Hip Angle: {far_hip_angle1:.1f}")
                st.write(
                    f"Left Far Hip Angle: {far_hip_angle2:.1f}" if side2 == "left" else f"Right Far Hip Angle: {far_hip_angle2:.1f}")

                # CSV + PDF export (same as before)
                now = datetime.now()
                df = pd.DataFrame([{
                    "Name": username,
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "left_jurdan_angle": round(left_angle, 1),
                    "right_jurdan_angle": round(right_angle, 1),
                    "left_flexion": round(flex1 if side1 == 'left' else flex2, 1),
                    "right_flexion": round(flex2 if side2 == 'right' else flex1, 1),
                    "left_extension": round(ext1 if side1 == 'right' else ext2, 1),
                    "right_extension": round(ext2 if side2 == 'right' else ext1, 1),
                    "jurdan_diff": round(diff, 1)
                }])
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Comparison CSV",
                                   data=csv,
                                   file_name=f"{username}_jurdan_comparison_{now.strftime('%Y%m%d_%H%M%S')}.csv",
                                   mime="text/csv")

                from fpdf import FPDF
                _, img1_buf = cv2.imencode(".jpg", img1)
                _, img2_buf = cv2.imencode(".jpg", img2)
                img1_bytes = img1_buf.tobytes()
                img2_bytes = img2_buf.tobytes()

                pdf = FPDF(orientation='L', unit='mm', format='A4')
                pdf.add_page()
                pdf.set_font("Arial", "B", 24)
                pdf.cell(0, 10, f"{username} - Jurdan Angle Comparison", ln=True, align="C")

                w_half = 140
                img1_path = os.path.join(tempfile.gettempdir(), "img1_temp.jpg")
                img2_path = os.path.join(tempfile.gettempdir(), "img2_temp.jpg")
                with open(img1_path, "wb") as f: f.write(img1_bytes)
                with open(img2_path, "wb") as f: f.write(img2_bytes)

                pdf.image(img1_path, x=10, y=30, w=w_half)
                pdf.image(img2_path, x=150, y=30, w=w_half)

                pdf.set_xy(10, 135)
                pdf.set_font("Arial", size=16)
                pdf.cell(w_half, 10, f"Left Jurdan Angle: {left_angle:.1f}", align="C")

                pdf.set_xy(150, 135)
                pdf.cell(w_half, 10, f"Right Jurdan Angle: {right_angle:.1f}", align="C")

                pdf.set_xy(10, 155)
                pdf.set_font("Arial", "B", 18)
                pdf.cell(0, 10, f"Difference: {diff:.1f}", ln=True, align="C")

                pdf_bytes = pdf.output(dest='S').encode('latin1')
                st.download_button("Download PDF Report",
                                   data=pdf_bytes,
                                   file_name=f"{username}_HipCheck_report.pdf",
                                   mime="application/pdf")
            else:
                st.warning("Could not determine both left and right Jurdan Angles.")
    else:
        st.warning("Please upload 1 or 2 images only.")
