"""Streamlit Web Application: 10-Class AI Object Recognition & Dataset Management System."""

import json
from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st

from src import config
from src.dataset import audit_and_clean_dataset
from scripts.download_images import download_dataset


st.set_page_config(
    page_title="AI Object Recognition System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished UI
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .prediction-badge {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1a73e8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_cached_predictor():
    """Lazily load predict_image function to prevent loading tensorflow if only browsing dataset."""
    try:
        from src.predict import predict_image, get_model
        return predict_image, get_model
    except Exception as e:
        return None, str(e)


# Header
st.markdown('<div class="main-title">🔍 AI Everyday Object Recognition System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">10-Class Image Classification powered by MobileNetV2 Transfer Learning and Wikimedia Commons</div>',
    unsafe_allow_html=True,
)

# Main Navigation Tabs
tab_inference, tab_dashboard, tab_metrics = st.tabs([
    "📸 Live Object Recognition",
    "📊 Dataset Management Dashboard",
    "📈 Model Evaluation & Reports",
])

# ==============================================================================
# TAB 1: OBJECT RECOGNITION (INFERENCE)
# ==============================================================================
with tab_inference:
    st.subheader("Recognize Everyday Objects")
    st.write(
        "Upload a photo or capture an image via webcam to classify it among the 10 target classes: "
        f"*{', '.join(config.DISPLAY_NAMES.values())}*."
    )

    if not config.BEST_MODEL_PATH.exists():
        st.warning(
            f"⚠️ Model weights not found at `{config.BEST_MODEL_PATH}`. "
            "Please train the model first by running `python -m src.train` or via the terminal."
        )

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        input_mode = st.radio("Choose Input Method:", ["File Upload", "Webcam Capture"], horizontal=True)
        img_to_predict = None

        if input_mode == "File Upload":
            uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file is not None:
                try:
                    img_to_predict = Image.open(uploaded_file)
                    st.image(img_to_predict, caption="Uploaded Image", use_container_width=True)
                except Exception as e:
                    st.error(f"Error opening image: {e}")
        else:
            cam_image = st.camera_input("Capture a live photo using your webcam")
            if cam_image is not None:
                try:
                    img_to_predict = Image.open(cam_image)
                    st.image(img_to_predict, caption="Webcam Capture", use_container_width=True)
                except Exception as e:
                    st.error(f"Error capturing image: {e}")

    with col_result:
        st.subheader("Prediction Results")
        if img_to_predict is not None:
            if not config.BEST_MODEL_PATH.exists():
                st.error("Cannot predict: Model is not yet trained. Run training first.")
            else:
                predict_fn, model_err = load_cached_predictor()
                if predict_fn is None:
                    st.error(f"Failed to load prediction engine: {model_err}")
                else:
                    with st.spinner("Analyzing image features with MobileNetV2..."):
                        try:
                            result = predict_fn(img_to_predict)

                            st.markdown(
                                f"""
                                <div class="metric-card" style="border-left: 5px solid #1a73e8;">
                                    <div style="font-size: 0.9rem; color: #5f6368; text-transform: uppercase;">Top Prediction</div>
                                    <div class="prediction-badge">{result['display_name']}</div>
                                    <div style="font-size: 1.2rem; color: #202124; font-weight: 500;">
                                        Confidence: <b>{result['confidence_percent']}%</b>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            if result["is_low_confidence"]:
                                st.warning(f"⚠️ {result['warning_message']}")

                            st.markdown("#### Top 3 Candidate Classes")
                            for rank, item in enumerate(result["top_predictions"], 1):
                                st.write(f"**#{rank} {item['display_name']}** — {item['confidence_percent']}%")
                                st.progress(float(item["confidence"]))

                            with st.expander("Show all 10 class probabilities"):
                                df_all = pd.DataFrame(
                                    [
                                        {"Class": config.DISPLAY_NAMES[k], "Probability": f"{v * 100:.2f}%"}
                                        for k, v in result["all_probabilities"].items()
                                    ]
                                )
                                st.dataframe(df_all, use_container_width=True)

                        except Exception as pred_err:
                            st.error(f"Prediction failed: {pred_err}")
        else:
            st.info("👈 Upload an image or capture a webcam photo to see predictions.")

# ==============================================================================
# TAB 2: DATASET DASHBOARD
# ==============================================================================
with tab_dashboard:
    st.subheader("Dataset Management & Audit Dashboard")
    st.write(
        "Monitor image counts, identify corrupt or duplicate files, and fetch missing images directly from Wikimedia Commons."
    )

    stats = audit_and_clean_dataset()
    total_valid = stats["total_valid"]
    target_total = config.TOTAL_TARGET
    total_missing = max(0, target_total - total_valid)

    # Top KPI Metrics
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Valid Images", f"{total_valid} / {target_total}")
    kpi2.metric("Dataset Readiness", f"{(total_valid / target_total) * 100:.1f}%")
    kpi3.metric("Missing Images", f"{total_missing}")
    kpi4.metric("Corrupt Files Found", f"{stats['total_corrupt']}")
    kpi5.metric("Duplicates Filtered", f"{stats['total_duplicates']}")

    st.markdown("---")

    # Category status table and download action
    col_table, col_chart = st.columns([3, 2], gap="large")

    category_rows = []
    for c_name in config.CLASS_NAMES:
        c_info = stats["categories"].get(c_name, {"valid": 0, "corrupt": 0, "duplicates": 0, "target": 50})
        v_count = c_info["valid"]
        missing = max(0, config.TARGET_PER_CLASS - v_count)
        status_badge = "✅ Ready" if v_count >= config.TARGET_PER_CLASS else f"⚠️ Missing ({missing})"
        category_rows.append(
            {
                "Category": config.DISPLAY_NAMES.get(c_name, c_name),
                "Internal Key": c_name,
                "Valid Images": v_count,
                "Target": config.TARGET_PER_CLASS,
                "Missing": missing,
                "Status": status_badge,
                "Duplicates": c_info["duplicates"],
                "Corrupt": c_info["corrupt"],
            }
        )
    df_cat = pd.DataFrame(category_rows)

    with col_table:
        st.markdown("#### Category Breakdown")
        st.dataframe(df_cat[["Category", "Valid Images", "Target", "Missing", "Status"]], use_container_width=True)

        st.markdown("#### Download Actions")
        if st.button("⬇️ Download Missing Images from Wikimedia Commons", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def streamlit_progress(c_name, current, target, action):
                status_text.write(f"Processing **{c_name}**: {action} ({current}/{target})")

            with st.spinner("Downloading images with perceptual deduplication and validation..."):
                try:
                    summary = download_dataset(
                        target_per_class=config.TARGET_PER_CLASS,
                        overwrite=False,
                        progress_callback=streamlit_progress,
                    )
                    progress_bar.progress(1.0)
                    st.success("Download routine finished successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Download encountered an issue: {e}")

    with col_chart:
        st.markdown("#### Images per Category Distribution")
        chart_data = df_cat.set_index("Category")[["Valid Images"]]
        st.bar_chart(chart_data)

    # Attribution view
    st.markdown("---")
    st.markdown("#### Dataset Attribution & Licensing (Wikimedia Commons)")
    if config.ATTRIBUTION_CSV.exists():
        try:
            df_attr = pd.read_csv(config.ATTRIBUTION_CSV)
            st.write(f"Total attribution records: **{len(df_attr)}**")
            st.dataframe(
                df_attr[["class", "filename", "author", "license", "source_page"]].head(25),
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Could not load attribution CSV: {e}")
    else:
        st.info("No attribution CSV found yet. Run the downloader to populate image metadata.")

# ==============================================================================
# TAB 3: MODEL EVALUATION & REPORTS
# ==============================================================================
with tab_metrics:
    st.subheader("Model Performance & Comprehensive Evaluation")
    st.write("Performance evaluation on the independent 15% stratified test split.")

    metrics_file = config.METRICS_JSON_PATH
    if metrics_file.exists():
        try:
            with metrics_file.open("r", encoding="utf-8") as f:
                metrics_data = json.load(f)

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            test_acc = metrics_data.get("test_accuracy", 0.0)
            macro_avg = metrics_data.get("macro_avg", {})

            m_col1.metric("Actual Test Accuracy", f"{test_acc * 100:.2f}%")
            m_col2.metric("Macro Precision", f"{macro_avg.get('precision', 0.0) * 100:.2f}%")
            m_col3.metric("Macro Recall", f"{macro_avg.get('recall', 0.0) * 100:.2f}%")
            m_col4.metric("Macro F1-Score", f"{macro_avg.get('f1-score', 0.0) * 100:.2f}%")

            st.markdown("---")

            col_cm, col_hist = st.columns([1, 1], gap="large")

            with col_cm:
                st.markdown("#### Confusion Matrix")
                if config.CONFUSION_MATRIX_PATH.exists():
                    st.image(str(config.CONFUSION_MATRIX_PATH), caption="Confusion Matrix on Test Set", use_container_width=True)
                else:
                    st.info("Confusion matrix plot not found.")

            with col_hist:
                st.markdown("#### Training & Validation History")
                if config.TRAINING_HISTORY_PATH.exists():
                    st.image(str(config.TRAINING_HISTORY_PATH), caption="Accuracy and Loss Curves", use_container_width=True)
                else:
                    st.info("Training history plot not found.")

            st.markdown("#### Per-Class Performance Metrics")
            per_class_data = metrics_data.get("per_class", {})
            if per_class_data:
                rows = []
                for k, v in per_class_data.items():
                    rows.append(
                        {
                            "Class": config.DISPLAY_NAMES.get(k, k),
                            "Precision": f"{v.get('precision', 0.0) * 100:.1f}%",
                            "Recall": f"{v.get('recall', 0.0) * 100:.1f}%",
                            "F1-Score": f"{v.get('f1-score', 0.0) * 100:.1f}%",
                            "Test Samples": int(v.get("support", 0)),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

        except Exception as e:
            st.error(f"Error loading evaluation report: {e}")
    else:
        st.info(
            "Evaluation metrics have not been generated yet. "
            "Train and evaluate the model using `python -m src.train` or `python -m src.evaluate`."
        )

st.markdown("---")
st.caption("AI Object Recognition System • MobileNetV2 Transfer Learning • Arch Linux / Caelestia Environment")
