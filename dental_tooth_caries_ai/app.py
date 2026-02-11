#!/usr/bin/env python3
"""
Streamlit demo for Dental Tooth Caries AI.

Upload a panoramic or bitewing dental image, run YOLO inference,
apply tooth-level assignment, and display annotated results.

Usage:
    streamlit run dental_tooth_caries_ai/app.py
"""

import os
import sys
import tempfile

import numpy as np
import streamlit as st
from PIL import Image

# ── Page config ──
st.set_page_config(
    page_title="🦷 Dental Caries AI",
    page_icon="🦷",
    layout="wide",
)

# ── Sidebar ──
st.sidebar.title("🦷 Dental Caries AI")
st.sidebar.markdown(
    "> **⚠️ Research Prototype**  \n"
    "> NOT for clinical diagnosis."
)

modality = st.sidebar.selectbox(
    "Modality",
    ["Panoramic (DENTEX)", "Bitewing"],
    index=0,
)

# Model weights
default_weights = {
    "Panoramic (DENTEX)": "runs/dental/pano_caries_only/weights/best.pt",
    "Bitewing": "runs/dental/bitewing_caries_only/weights/best.pt",
}
weights_path = st.sidebar.text_input(
    "Model Weights",
    value=default_weights.get(modality, ""),
    help="Path to trained .pt weights file",
)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0, max_value=1.0, value=0.25, step=0.05,
)

tooth_assign = False
if modality == "Bitewing":
    tooth_assign = st.sidebar.checkbox(
        "🧪 Experimental: Tooth Assignment",
        value=False,
        help="Map lesion detections to tooth proposals (experimental baseline)",
    )

# Class names
CLASS_NAMES = {
    "Panoramic (DENTEX)": ["Caries", "Deep Caries"],
    "Bitewing": ["caries"],
}

# ── Main content ──
st.title("🦷 Dental Tooth Caries AI — Demo")
st.markdown(
    "Upload a dental radiograph to detect caries with **tooth-level** annotations."
)

uploaded = st.file_uploader(
    "Upload Dental Image",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Supported: panoramic X-ray or bitewing radiograph",
)

if uploaded is not None:
    # Load image
    pil_img = Image.open(uploaded).convert("RGB")
    img_array = np.array(pil_img)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original")
        st.image(pil_img, use_container_width=True)

    # Run inference
    if not os.path.isfile(weights_path):
        st.error(
            f"Model weights not found: `{weights_path}`  \n"
            f"Train a model first with `make train MODALITY=...`"
        )
        st.stop()

    with st.spinner("Running inference ..."):
        try:
            from ultralytics import YOLO
            import cv2

            from dental_tooth_caries_ai.tooth_level.assign_lesions_to_teeth import (
                assign_lesions_to_teeth,
                make_direct_tooth_predictions,
            )
            from dental_tooth_caries_ai.tooth_level.render_overlays import render_overlay
            from dental_tooth_caries_ai.tooth_level.tooth_proposals import (
                propose_teeth_heuristic,
            )

            model = YOLO(weights_path)

            # Convert RGB to BGR for OpenCV / YOLO
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            results = model(img_bgr, conf=conf_threshold)

            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                class_names = CLASS_NAMES.get(modality, ["unknown"])

                detections = []
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    detections.append({
                        "bbox": (float(xyxy[0]), float(xyxy[1]),
                                 float(xyxy[2]), float(xyxy[3])),
                        "class_id": int(box.cls[0]),
                        "confidence": float(box.conf[0]),
                    })

                if modality == "Panoramic (DENTEX)":
                    # Direct tooth-level predictions
                    tooth_preds = make_direct_tooth_predictions(
                        detections, class_names,
                    )
                    annotated = render_overlay(img_bgr, tooth_preds)

                elif modality == "Bitewing" and tooth_assign:
                    # Experimental: heuristic tooth proposals + assignment
                    teeth = propose_teeth_heuristic(img_bgr)
                    lesion_boxes = [d["bbox"] for d in detections]
                    lesion_classes = [
                        class_names[d["class_id"]]
                        if d["class_id"] < len(class_names)
                        else "unknown"
                        for d in detections
                    ]
                    lesion_confs = [d["confidence"] for d in detections]

                    tooth_preds = assign_lesions_to_teeth(
                        teeth, lesion_boxes, lesion_classes, lesion_confs,
                    )
                    annotated = render_overlay(
                        img_bgr, tooth_preds,
                        all_teeth=teeth,
                        show_lesion_boxes=True,
                    )
                else:
                    # Bitewing without tooth assignment — show lesion boxes only
                    tooth_preds = make_direct_tooth_predictions(
                        detections, class_names,
                    )
                    annotated = render_overlay(img_bgr, tooth_preds)

                # Convert back to RGB for display
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                with col2:
                    st.subheader("🔍 Detection Results")
                    st.image(annotated_rgb, use_container_width=True)

                # Results table
                st.subheader("📊 Detections")
                rows = []
                for p in tooth_preds:
                    rows.append({
                        "Tooth ID": p.tooth.tooth_id,
                        "Diagnosis": p.diagnosis,
                        "Confidence": f"{p.confidence:.1%}",
                        "Source": p.source,
                        "BBox": f"({p.tooth.bbox[0]:.0f}, {p.tooth.bbox[1]:.0f}, "
                                f"{p.tooth.bbox[2]:.0f}, {p.tooth.bbox[3]:.0f})",
                    })

                if rows:
                    st.table(rows)
                    st.success(f"Found {len(rows)} detection(s).")
                else:
                    st.info("No detections above confidence threshold.")

            else:
                with col2:
                    st.subheader("🔍 Detection Results")
                    st.info("No detections found. Try lowering the confidence threshold.")

        except Exception as e:
            st.error(f"Inference failed: {e}")
            import traceback
            st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown(
    "⚠️ **Research Prototype** — NOT for clinical diagnosis. "
    "Do not use predictions from this system for treatment decisions. "
    "Built with [Ultralytics YOLOv8](https://docs.ultralytics.com/) "
    "and [YOLO Training Template](https://github.com/computer-vision-with-marco/yolo-training-template)."
)
