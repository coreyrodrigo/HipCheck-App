import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import tempfile
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ========== MODEL SETUP ==========
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

# ========== ANGLE UTILS ==========
def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

# ========== OPENCV MANUAL ADJUSTER ==========
clicked_points = []
selected_idx = None

def mouse_event(event, x, y, flags, param):
    global clicked_points, selected_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        for idx, (px, py) in enumerate(clicked_points):
            if abs(px - x) < 15 and abs(py - y) < 15:
                selected_idx = idx
    elif event == cv2.EVENT_MOUSEMOVE and selected_idx is not None:
        clicked_points[selected_idx] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        selected_idx = None

def manual_edit(image, points):
    global clicked_points
    clicked_points = points.copy()

    cv2.namedWindow("Adjust Points")
    cv2.setMouseCallback("Adjust Points", mouse_event)

    while True:
        temp = image.copy()
        for pt in clicked_points:
            cv2.circle(temp, pt, 10, (0, 0, 255), -1)
        cv2.line(temp, clicked_points[0], clicked_points[1], (0, 255, 255), 4)
        cv2.line(temp, clicked_points[1], clicked_points[2], (0, 255, 255), 4)
        cv2.line(temp, clicked_points[2], clicked_points[3], (0, 255, 255), 4)

        hip_angle = calc_angle(clicked_points[0], clicked_points[1], clicked_points[2])
        knee_angle = calc_angle(clicked_points[1], clicked_points[2], clicked_points[3])
        jurdan_angle = (180 - hip_angle) + (knee_angle - 90)

        # Text Labels
        cv2.putText(temp, f"Hip: {hip_angle:.1f}", (clicked_points[1][0] + 10, clicked_points[1][1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(temp, f"Knee: {knee_angle:.1f}", (clicked_points[2][0] + 10, clicked_points[2][1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # === Jurdan Angle Over HIP ===
        ja_text = f"Jurdan Angle: {jurdan_angle:.1f}"
        text_size = cv2.getTextSize(ja_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        ja_x = clicked_points[1][0] - text_size[0] // 2
        ja_y = clicked_points[1][1] - 60
        cv2.rectangle(temp, (ja_x - 10, ja_y - 30), (ja_x + text_size[0] + 10, ja_y + 10), (0, 0, 0), -1)
        cv2.putText(temp, ja_text, (ja_x, ja_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

        cv2.imshow("Adjust Points", temp)
        if cv2.waitKey(20) & 0xFF == 13:  # Enter to finish
            break

    cv2.destroyAllWindows()
    return clicked_points, hip_angle, knee_angle, jurdan_angle, temp

# ========== IMAGE PROCESSING ==========
def process_image(image_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(image_file.read())
    image_bgr = cv2.imread(temp_file.name)
    image = mp.Image.create_from_file(temp_file.name)
    model = load_model()
    results = model.detect(image)

    if not results.pose_landmarks:
        return None, None, None, None, None

    landmarks = results.pose_landmarks[0]
    h, w = image_bgr.shape[:2]

    def to_px(idx): return int(landmarks[idx].x * w), int(landmarks[idx].y * h)

    shoulder = to_px(11)  # trunk
    hip = to_px(23)
    knee = to_px(25)
    ankle = to_px(27)

    initial_points = [shoulder, hip, knee, ankle]

    st.info("Manual adjustment window will open. Press ENTER when done.")
    final_points, hip_angle, knee_angle, jurdan_angle, edited_img = manual_edit(image_bgr, initial_points)

    return edited_img, final_points, hip_angle, knee_angle, jurdan_angle

# ========== STREAMLIT UI ==========
st.set_page_config("Jurdan Angle Manual Editor", layout="centered")
st.title("Jurdan Angle – Manual Joint Editor")

uploaded_file = st.file_uploader("Upload image (supine pose side view)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    edited_img, pts, hip_angle, knee_angle, jurdan_angle = process_image(uploaded_file)
    if edited_img is not None:
        st.image(cv2.cvtColor(edited_img, cv2.COLOR_BGR2RGB), channels="RGB", caption="Edited Pose", use_container_width=True)

        st.markdown("### Final Angles")
        st.write(f"**Hip Angle**: {hip_angle:.1f}°")
        st.write(f"**Knee Angle**: {knee_angle:.1f}°")
        st.write(f"**Jurdan Angle**: {jurdan_angle:.1f}°")

        df = pd.DataFrame([{
            "hip_angle": round(hip_angle, 1),
            "knee_angle": round(knee_angle, 1),
            "jurdan_angle": round(jurdan_angle, 1),
        }])
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Angles as CSV", csv, file_name="jurdan_angles.csv", mime="text/csv")
    else:
        st.error("Pose not detected.")
