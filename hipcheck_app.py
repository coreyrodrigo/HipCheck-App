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

def create_adjustment_interface(image_bgr, landmarks, width, height, image_key):
    """
    Create a simple form-based interface for adjusting joint positions
    """
    
    JOINTS = {
        "left_shoulder": 11, "right_shoulder": 12,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28,
        "left_foot": 31, "right_foot": 32
    }
    
    # Show the original image with detected joints
    display_img = image_bgr.copy()
    
    # Draw original detection
    for name, idx in JOINTS.items():
        pos = (int(landmarks[idx].x * width), int(landmarks[idx].y * height))
        cv2.circle(display_img, pos, 8, (0, 255, 0), -1)  # Green circles
        cv2.putText(display_img, name.replace('_', ' '), (pos[0] + 12, pos[1] - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), 
             caption="AI Detected Joints (Green) - Adjust if needed below", 
             use_container_width=True)
    
    # Create adjustment form
    st.markdown("#### 🎯 Manual Joint Position Adjustment")
    st.markdown("*Adjust coordinates only for joints that appear incorrectly positioned*")
    
    with st.form(f"adjust_joints_{image_key}"):
        adjustments = {}
        
        # Create expandable sections for each side
        left_expander = st.expander("🔧 Adjust Left Side Joints", expanded=False)
        with left_expander:
            for joint_name in ["left_hip", "left_knee", "left_shoulder", "left_ankle", "left_foot"]:
                idx = JOINTS[joint_name]
                original_x = int(landmarks[idx].x * width)
                original_y = int(landmarks[idx].y * height)
                
                st.markdown(f"**{joint_name.replace('_', ' ').title()}** (Original: {original_x}, {original_y})")
                
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    new_x = st.number_input(f"X coordinate", 
                                          min_value=0, max_value=width, value=original_x,
                                          key=f"{joint_name}_x_{image_key}")
                with col2:
                    new_y = st.number_input(f"Y coordinate", 
                                          min_value=0, max_value=height, value=original_y,
                                          key=f"{joint_name}_y_{image_key}")
                with col3:
                    changed = (new_x != original_x) or (new_y != original_y)
                    if changed:
                        st.success("✏️")
                    else:
                        st.info("📍")
                
                if changed:
                    adjustments[joint_name] = {
                        "x": new_x, "y": new_y,
                        "normalized_x": new_x / width,
                        "normalized_y": new_y / height
                    }
        
        right_expander = st.expander("🔧 Adjust Right Side Joints", expanded=False)
        with right_expander:
            for joint_name in ["right_hip", "right_knee", "right_shoulder", "right_ankle", "right_foot"]:
                idx = JOINTS[joint_name]
                original_x = int(landmarks[idx].x * width)
                original_y = int(landmarks[idx].y * height)
                
                st.markdown(f"**{joint_name.replace('_', ' ').title()}** (Original: {original_x}, {original_y})")
                
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    new_x = st.number_input(f"X coordinate", 
                                          min_value=0, max_value=width, value=original_x,
                                          key=f"{joint_name}_x_{image_key}")
                with col2:
                    new_y = st.number_input(f"Y coordinate", 
                                          min_value=0, max_value=height, value=original_y,
                                          key=f"{joint_name}_y_{image_key}")
                with col3:
                    changed = (new_x != original_x) or (new_y != original_y)
                    if changed:
                        st.success("✏️")
                    else:
                        st.info("📍")
                
                if changed:
                    adjustments[joint_name] = {
                        "x": new_x, "y": new_y,
                        "normalized_x": new_x / width,
                        "normalized_y": new_y / height
                    }
        
        submitted = st.form_submit_button("🔄 Apply Adjustments and Recalculate", type="primary")
        
        if submitted:
            st.session_state[f"adjustments_{image_key}"] = adjustments
            if adjustments:
                st.success(f"✅ Applied {len(adjustments)} joint adjustments!")
            else:
                st.info("ℹ️ No adjustments made - using original AI predictions")
            st.rerun()
    
    return st.session_state.get(f"adjustments_{image_key}", {})

def apply_adjustments_to_landmarks(landmarks, adjustments):
    """
    Apply manual adjustments to MediaPipe landmarks
    """
    if not adjustments:
        return landmarks
    
    JOINTS = {
        "left_shoulder": 11, "right_shoulder": 12,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28,
        "left_foot": 31, "right_foot": 32
    }
    
    # Create a copy of landmarks
    adjusted_landmarks = []
    
    for i, landmark in enumerate(landmarks):
        # Check if this landmark needs adjustment
        joint_name = None
        for name, idx in JOINTS.items():
            if idx == i:
                joint_name = name
                break
        
        if joint_name and joint_name in adjustments:
            # Create adjusted landmark
            adj = adjustments[joint_name]
            
            class AdjustedLandmark:
                def __init__(self, x, y, z, visibility, presence):
                    self.x = x
                    self.y = y
                    self.z = z
                    self.visibility = visibility
                    self.presence = presence
            
            adjusted_landmark = AdjustedLandmark(
                x=adj["normalized_x"],
                y=adj["normalized_y"],
                z=landmark.z,
                visibility=landmark.visibility,
                presence=landmark.presence
            )
            adjusted_landmarks.append(adjusted_landmark)
        else:
            adjusted_landmarks.append(landmark)
    
    return adjusted_landmarks

def process_image(image_file, manual_adjustments=None):
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
    
    # Apply manual adjustments if provided
    if manual_adjustments:
        landmarks = apply_adjustments_to_landmarks(landmarks, manual_adjustments)

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
        far_hip_angle = calc_angle(get_xy(12), get_xy(24), get_xy(26))
        far_knee_angle = calc_angle(get_xy(24), get_xy(26), get_xy(28))
        close_hip_px = to_px(JOINTS["left_hip"])
        far_hip_px = to_px(JOINTS["right_hip"])
        far_knee_px = to_px(JOINTS["right_knee"])
    else:
        close_hip_angle = calc_angle(get_xy(12), get_xy(24), get_xy(26))
        far_hip_angle = calc_angle(get_xy(11), get_xy(23), get_xy(25))
        far_knee_angle = calc_angle(get_xy(23), get_xy(25), get_xy(27))
        close_hip_px = to_px(JOINTS["right_hip"])
        far_hip_px = to_px(JOINTS["left_hip"])
        far_knee_px = to_px(JOINTS["left_knee"])

    close_hip_flexion = 180 - close_hip_angle
    far_hip_flexion = 180 - far_hip_angle
    far_knee_extension = far_knee_angle - 90
    jurdan_angle = close_hip_flexion + far_knee_extension
    hipcheck_angle = jurdan_angle - (90-far_hip_flexion)

    # === Enhanced Drawing with adjustment indicators ===
    def draw_joint_line(a, b):
        cv2.line(image_bgr, to_px(JOINTS[a]), to_px(JOINTS[b]), (0, 255, 255), 6)

    for pair in [("left_shoulder", "left_hip"), ("left_hip", "left_knee"),
                 ("left_knee", "left_ankle"), ("left_ankle", "left_foot"),
                 ("right_shoulder", "right_hip"), ("right_hip", "right_knee"),
                 ("right_knee", "right_ankle"), ("right_ankle", "right_foot")]:
        draw_joint_line(*pair)

    # Draw joints with different colors for adjusted vs original
    for name, idx in JOINTS.items():
        pos = to_px(idx)
        if manual_adjustments and name in manual_adjustments:
            # Orange for manually adjusted joints
            cv2.circle(image_bgr, pos, 12, (0, 165, 255), -1)
            cv2.circle(image_bgr, pos, 12, (255, 255, 255), 3)
            # Add "M" indicator
            cv2.putText(image_bgr, "M", (pos[0] - 4, pos[1] + 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            # Red for AI-detected joints
            cv2.circle(image_bgr, pos, 10, (0, 0, 255), -1)
            cv2.circle(image_bgr, pos, 10, (255, 255, 255), 2)
