import hashlib
import html
import io
import json
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from inference import predict_crop, predict_fusion


st.set_page_config(
    page_title="Verdant | Soybean Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

LABELS = {
    "healthy": "Healthy",
    "mosaic": "Mosaic",
    "rust": "Rust",
    "pest_attack": "Pest attack"
}

MODELS = {
    "leaf": "ConvNeXt-Tiny",
    "drone": "MaxViT-Tiny"
}

COLORS = {
    "healthy": "#6EE7B7",
    "mosaic": "#F0CD79",
    "rust": "#F2A18A",
    "pest_attack": "#B8AAFA"
}

PRECAUTIONS = {
    "healthy": [
        "Continue inspecting representative plants across the plot.",
        "If visible symptoms are present, do not dismiss them because "
        "the model predicted healthy.",
        "Record changes over time and seek assessment if symptoms "
        "persist or spread."
    ],
    "mosaic": [
        "Inspect additional plants and take clear close-up photographs "
        "of the affected and surrounding leaves.",
        "Ask a local agricultural extension officer or plant diagnostic "
        "service to distinguish mosaic-like symptoms from other causes.",
        "If soybean mosaic virus is confirmed, discuss appropriate "
        "seed-health and resistant-variety options for future planting."
    ],
    "rust": [
        "Inspect affected plants at ground level and photograph both "
        "surfaces of symptomatic leaves.",
        "Arrange local diagnostic confirmation; a leaf-spot appearance "
        "alone does not establish rust.",
        "Discuss management with a local crop adviser using confirmed "
        "diagnosis, crop stage, field conditions and local guidance."
    ],
    "pest_attack": [
        "Inspect several locations across the field for live insects "
        "and fresh feeding damage.",
        "Assess damage across the canopy, rather than using one heavily "
        "damaged leaf as the field-wide estimate.",
        "Identify the pest and crop growth stage before discussing "
        "locally appropriate action thresholds with a crop adviser."
    ]
}

GUIDANCE_LINKS = {
    "healthy": (
        "Soybean pest-management resources",
        "https://extension.umn.edu/soybean/soybean-pest-management"
    ),
    "mosaic": (
        "Soybean mosaic virus — University of Minnesota Extension",
        "https://extension.umn.edu/agriculture/crop-production/"
        "soybean/soybean-mosaic-virus"
    ),
    "rust": (
        "Soybean disease diagnosis and scouting — Iowa State Extension",
        "https://crops.extension.iastate.edu/"
        "soybean-diseases-symptoms-scouting-and-management"
    ),
    "pest_attack": (
        "Soybean defoliator scouting — Iowa State Extension",
        "https://crops.extension.iastate.edu/"
        "cropnews/2023/07/keep-eye-out-soybean-defoliators"
    )
}

CSS = """
<style>
:root {
    --bg: #080f13;
    --panel: #101b21;
    --line: #26363e;
    --text: #eaf2f5;
    --muted: #a5bac4;
    --green: #6ee7b7;
}
.stApp {
    background:
        radial-gradient(ellipse at 85% 0%, #102b26 0%, transparent 42%),
        #080f13;
    color: var(--text);
}
[data-testid="stHeader"] {
    background: rgba(8, 15, 19, .92);
}
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background: #0c161c;
    border-right: 1px solid #223139;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    letter-spacing: -.035em;
}
p, label {
    line-height: 1.6;
}
[data-testid="stCaptionContainer"] {
    color: #a5bac4;
}
.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 4px 0 28px;
}
.brand-icon {
    display: grid;
    place-items: center;
    width: 44px;
    height: 44px;
    border: 1px solid #3d6959;
    background: #15362b;
    border-radius: 14px;
    color: #99f1c9;
    font-size: 25px;
}
.brand-name {
    font-size: 23px;
    font-weight: 800;
    letter-spacing: -.8px;
    color: #f3faf7;
}
.brand-sub {
    color: #9ab2bd;
    font-size: 10px;
    letter-spacing: 2px;
}
.side-label, .eyebrow {
    color: #8eacb9;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.side-card {
    padding: 18px;
    background: #122129;
    border: 1px solid #2b3e48;
    border-radius: 15px;
    margin-top: 20px;
}
.side-card strong {
    display: block;
    margin-top: 8px;
    font-size: 17px;
    color: #ecf8f2;
}
.side-card p {
    color: #acc1cb;
    font-size: 12px;
    margin-bottom: 0;
}
.topline {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    margin-bottom: 24px;
    color: #adc1ca;
    font-size: 12px;
}
.pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid #345448;
    background: #122820;
    color: #b7eed2;
    border-radius: 30px;
    padding: 6px 12px;
    font-size: 11px;
    white-space: nowrap;
}
.dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #6ee7b7;
}
.hero {
    position: relative;
    overflow: hidden;
    padding: 36px 38px;
    border: 1px solid #2a4943;
    border-radius: 24px;
    background: linear-gradient(115deg, #142d28, #101c24 75%);
    margin-bottom: 22px;
    animation: enter .55s ease both;
}
.hero::after {
    content: "";
    position: absolute;
    right: -50px;
    top: -90px;
    width: 360px;
    height: 360px;
    border: 1px solid #315147;
    border-radius: 50%;
    box-shadow:
        0 0 0 36px rgba(110,231,183,.025),
        0 0 0 72px rgba(110,231,183,.018);
    pointer-events: none;
}
.hero .eyebrow {
    color: #a8e1c6;
    position: relative;
    z-index: 1;
}
.hero h1 {
    font-size: clamp(30px, 3.5vw, 48px);
    line-height: 1.12;
    color: #f2f8f5;
    margin: 14px 0 14px;
    max-width: 750px;
    position: relative;
    z-index: 1;
}
.hero h1 span {
    color: #8ce6bc;
}
.hero p {
    max-width: 650px;
    color: #bad0d3;
    font-size: 15px;
    position: relative;
    z-index: 1;
    margin-bottom: 20px;
}
.hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    position: relative;
    z-index: 1;
}
.hero-tags span {
    color: #cadbdc;
    font-size: 11px;
    padding: 6px 11px;
    border: 1px solid #38514d;
    border-radius: 7px;
    background: rgba(8,15,19,.25);
}
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 28px;
}
.stat {
    border: 1px solid var(--line);
    border-radius: 15px;
    background: rgba(16,27,33,.85);
    padding: 18px 22px;
    transition: transform .2s, border-color .2s;
}
.stat:hover {
    transform: translateY(-3px);
    border-color: #537d6e;
}
.stat strong {
    color: #f0f7f4;
    font-size: 23px;
    display: block;
}
.stat span {
    font-size: 12px;
    color: #a5bac4;
}
[data-baseweb="tab-list"] {
    gap: 22px;
    border-bottom: 1px solid #26363e;
    margin-bottom: 22px;
}
[data-baseweb="tab"] {
    color: #a5bac4;
    height: 48px;
    font-weight: 600;
}
[data-baseweb="tab"][aria-selected="true"] {
    color: #9df0cd;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px !important;
}
.panel-head {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 8px;
}
.step-number {
    display: grid;
    place-items: center;
    width: 32px;
    height: 32px;
    border: 1px solid #365447;
    border-radius: 10px;
    color: #9de7c3;
    background: #142a23;
    font-size: 12px;
    font-weight: 700;
}
.panel-head h3 {
    font-size: 20px;
    margin: 0;
    color: #eff6f3;
}
[data-testid="stFileUploaderDropzone"] {
    background: #101f27;
    border: 1px dashed #52736a;
    border-radius: 15px;
    transition: background .2s, border-color .2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
    background: #152b2b;
    border-color: #8de7bf;
}
.stButton > button,
.stDownloadButton > button {
    border-radius: 11px;
    min-height: 44px;
    transition: transform .18s, box-shadow .18s, border-color .18s;
    font-weight: 600;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 7px 22px rgba(0,0,0,.2);
}
.stButton > button[kind="primary"] {
    background: #8de6bf;
    border: 1px solid #8de6bf;
    color: #092019;
}
.stButton > button[kind="primary"]:hover {
    background: #acf2d1;
    border-color: #acf2d1;
    color: #092019;
}
.stButton > button:disabled {
    transform: none;
    opacity: .45;
}
[data-testid="stImage"] img {
    border-radius: 14px;
}
.empty-result {
    min-height: 345px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    border: 1px dashed #30434d;
    background: radial-gradient(circle at center, #18332f, #101b21 68%);
    border-radius: 17px;
    padding: 32px;
}
.orbit {
    height: 94px;
    width: 94px;
    border: 1px solid #527b69;
    border-radius: 50%;
    display: grid;
    place-items: center;
    position: relative;
    margin-bottom: 22px;
    box-shadow: 0 0 0 13px rgba(110,231,183,.035);
}
.orbit::before {
    content: "";
    position: absolute;
    inset: -7px;
    border: 1px solid transparent;
    border-top-color: #8de6bf;
    border-radius: 50%;
    animation: spin 12s linear infinite;
}
.orbit span {
    color: #a8eac8;
    font-size: 36px;
}
.empty-result h3 {
    font-size: 20px;
    margin: 0 0 8px;
}
.empty-result p {
    max-width: 290px;
    color: #a5bac4;
    font-size: 13px;
}
.result-card {
    padding: 24px;
    border: 1px solid #456355;
    border-radius: 17px;
    background: linear-gradient(120deg, #18342b, #111e26);
    margin: 8px 0 20px;
    animation: enter .45s ease both;
}
.result-card h2 {
    color: #f0f8f3;
    font-size: 32px;
    margin: 10px 0 14px;
}
.result-metrics {
    display: flex;
    gap: 32px;
}
.result-metrics strong {
    display: block;
    font-size: 25px;
    color: #aff0ce;
}
.result-metrics span {
    font-size: 11px;
    color: #bbd0d1;
}
.score-row {
    margin: 15px 0;
}
.score-label {
    display: flex;
    justify-content: space-between;
    gap: 15px;
    margin-bottom: 8px;
    color: #d9e5e9;
    font-size: 13px;
}
.score-track {
    height: 8px;
    background: #25353d;
    border-radius: 10px;
    overflow: hidden;
}
.score-fill {
    height: 100%;
    border-radius: 10px;
    transform-origin: left;
    animation: fill .65s ease both;
}
.note {
    border-left: 2px solid #b7a16b;
    padding: 10px 14px;
    margin: 20px 0;
    background: #211f1a;
    color: #d8cfb8;
    font-size: 12px;
    border-radius: 0 8px 8px 0;
}
.fusion-heading {
    padding: 23px 25px;
    margin: 12px 0 18px;
    border: 1px solid #484365;
    border-radius: 18px;
    background: linear-gradient(120deg, #211f35, #111e26);
}
.fusion-heading .eyebrow {
    color: #c5bafa;
}
.fusion-heading h2 {
    color: #f0eef9;
    font-size: 27px;
    margin: 8px 0;
}
.fusion-heading p {
    color: #b9bdd0;
    font-size: 13px;
    margin-bottom: 0;
}
.guidance-card {
    border: 1px solid #344d46;
    background: #12231f;
    border-radius: 13px;
    padding: 17px 19px;
    margin-top: 15px;
}
.guidance-card h4 {
    color: #d5f3e3;
    font-size: 15px;
    margin: 0 0 10px;
}
.guidance-card li {
    color: #c4d7cf;
    font-size: 13px;
    line-height: 1.65;
    margin-bottom: 7px;
}
.guidance-card ul {
    margin: 0;
    padding-left: 18px;
}
.footer {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    border-top: 1px solid #26363e;
    padding-top: 18px;
    margin-top: 35px;
    color: #9bb0bc;
    font-size: 11px;
}
@keyframes enter {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}
@keyframes spin {
    to {transform: rotate(360deg);}
}
@keyframes fill {
    from {transform: scaleX(0);}
    to {transform: scaleX(1);}
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
    }
}
@media (max-width: 700px) {
    .hero {padding: 25px;}
    .hero::after {display: none;}
    .stat-grid {grid-template-columns: 1fr;}
    .topline {flex-wrap: wrap;}
    .result-metrics {gap: 18px; flex-wrap: wrap;}
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def prediction_lock():
    return threading.Lock()


def read_image(image_bytes):
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("Please use an image smaller than 20 MB.")

    with Image.open(io.BytesIO(image_bytes)) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def analyze_image(image_bytes, mode):
    image = read_image(image_bytes)

    with tempfile.TemporaryDirectory() as temporary_directory:
        image_path = Path(temporary_directory) / "image.png"
        image.save(image_path)

        with prediction_lock():
            result = predict_crop(image_path, mode)

    return result


def analyze_fusion(leaf_bytes, drone_bytes):
    leaf_image = read_image(leaf_bytes)
    drone_image = read_image(drone_bytes)

    with tempfile.TemporaryDirectory() as temporary_directory:
        leaf_path = Path(temporary_directory) / "leaf.png"
        drone_path = Path(temporary_directory) / "drone.png"

        leaf_image.save(leaf_path)
        drone_image.save(drone_path)

        with prediction_lock():
            result = predict_fusion(leaf_path, drone_path)

    return result


def panel_heading(number, title):
    st.markdown(
        f'<div class="panel-head">'
        f'<div class="step-number">{html.escape(number)}</div>'
        f'<h3>{html.escape(title)}</h3></div>',
        unsafe_allow_html=True
    )


def render_scores(scores):
    rows = []

    for label, score in sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    ):
        percentage = max(0.0, min(100.0, float(score) * 100))

        rows.append(
            f'<div class="score-row">'
            f'<div class="score-label">'
            f'<span>{LABELS[label]}</span>'
            f'<strong>{percentage:.2f}%</strong></div>'
            f'<div class="score-track">'
            f'<div class="score-fill" '
            f'style="width:{percentage:.6f}%;'
            f'background:{COLORS[label]}"></div>'
            f'</div></div>'
        )

    st.markdown("".join(rows), unsafe_allow_html=True)


def render_score_explanation(scores):
    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    top_label, top_score = ranked[0]
    next_label, next_score = ranked[1]
    margin = (float(top_score) - float(next_score)) * 100

    st.caption(
        f"Top score: {LABELS[top_label]} {top_score:.2%}. "
        f"Next: {LABELS[next_label]} {next_score:.2%}. "
        f"Difference: {margin:.2f} percentage points."
    )

    st.caption(
        "These percentages describe model preference, not the proportion "
        "of infected tissue or the severity of disease. No score threshold "
        "has been validated for treatment decisions."
    )


def render_precautions(label, aerial=False, disagreement=False):
    if label is None:
        st.info(
            "There is no unique combined category. Review the individual "
            "predictions and obtain clearer, verified observations."
        )
        return

    heading = f"Inspection guidance · possible {LABELS[label].lower()}"

    if label == "healthy":
        heading = "Inspection guidance · predicted healthy"

    bullet_html = "".join(
        f"<li>{html.escape(item)}</li>"
        for item in PRECAUTIONS[label]
    )

    st.markdown(
        f'<div class="guidance-card">'
        f'<h4>{html.escape(heading)}</h4>'
        f'<ul>{bullet_html}</ul></div>',
        unsafe_allow_html=True
    )

    if aerial:
        st.caption(
            "For an aerial prediction, inspect the corresponding plants "
            "at ground level before drawing disease conclusions."
        )

    if disagreement:
        st.caption(
            "The branches disagree. Treat category-specific guidance as "
            "conditional until symptoms are checked independently."
        )

    st.caption(
        "General scouting guidance only. Do not choose a pesticide, "
        "dosage or spray schedule from this prediction. Seek locally "
        "applicable advice from a qualified crop adviser or KVK."
    )

    title, address = GUIDANCE_LINKS[label]

    with st.expander("Guidance reference"):
        st.markdown(f"[{title}]({address})")
        st.caption(
            "Background reference for diagnosis and scouting; "
            "local recommendations and field conditions take precedence."
        )


def reset_fusion_inputs():
    st.session_state.verdant_fusion_result = None
    st.session_state.fusion_pair_confirmed = False


def reset_fusion_result():
    st.session_state.verdant_fusion_result = None


if "verdant_history" not in st.session_state:
    st.session_state.verdant_history = []

if "verdant_result" not in st.session_state:
    st.session_state.verdant_result = None

if "verdant_fusion_result" not in st.session_state:
    st.session_state.verdant_fusion_result = None

if "verdant_fusion_history" not in st.session_state:
    st.session_state.verdant_fusion_history = []

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">❧</div>
            <div>
                <div class="brand-name">verdant.</div>
                <div class="brand-sub">SOYBEAN INTELLIGENCE</div>
            </div>
        </div>
        <div class="side-label">Analysis workspace</div>
        """,
        unsafe_allow_html=True
    )

    selected_branch = st.radio(
        "Choose an image source",
        ["Leaf / smartphone", "Drone / aerial"],
        key="verdant_branch"
    )

    mode = "leaf" if selected_branch.startswith("Leaf") else "drone"

    st.markdown(
        f"""
        <div class="side-card">
            <div class="eyebrow">Active model</div>
            <strong>{MODELS[mode]}</strong>
            <p>{
                "Fine-scale leaf image classification"
                if mode == "leaf"
                else "Aerial soybean image classification"
            }</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown("**Image guidance**")

    if mode == "leaf":
        st.caption(
            "Use a clear soybean leaf photograph. "
            "Keep relevant symptoms visible and avoid screenshots "
            "with surrounding interface elements."
        )
    else:
        st.caption(
            "Use an actual aerial soybean image. "
            "A close-up leaf photo is not a drone test image."
        )

    st.caption("JPG · PNG · WEBP  |  Maximum upload: 20 MB")

    st.divider()
    st.markdown("**Supported categories**")
    st.caption("Healthy · Mosaic · Rust · Pest attack")
    st.caption(
        "Other crops and unrelated images are not automatically rejected."
    )

st.markdown(
    """
    <div class="topline">
        <span>WORKSPACE / CROP HEALTH ASSESSMENT</span>
        <span class="pill"><span class="dot"></span> Research demo</span>
    </div>
    <div class="hero">
        <div class="eyebrow">A closer look at crop health</div>
        <h1>Understand your crop.<br><span>One image at a time.</span></h1>
        <p>Explore soybean health with dedicated models for leaf
        photographs and aerial imagery, in one focused workspace.</p>
        <div class="hero-tags">
            <span>Leaf analysis</span>
            <span>Aerial analysis</span>
            <span>Downloadable results</span>
        </div>
    </div>
    <div class="stat-grid">
        <div class="stat"><strong>02</strong><span>Dedicated prediction branches</span></div>
        <div class="stat"><strong>04</strong><span>Supported soybean categories</span></div>
        <div class="stat"><strong>06</strong><span>Models evaluated during development</span></div>
    </div>
    """,
    unsafe_allow_html=True
)

workspace_tab, history_tab, evidence_tab = st.tabs([
    "Analysis studio",
    "Prediction history",
    "Models & evidence"
])

with workspace_tab:
    left_column, right_column = st.columns([1.05, 1], gap="large")

    with left_column:
        with st.container(border=True):
            panel_heading("01", "Your image")
            st.caption(
                f"Selected source: {selected_branch}. "
                "Choose an image to begin."
            )

            uploaded_file = st.file_uploader(
                "Upload a soybean image",
                type=["jpg", "jpeg", "png", "webp"],
                key="verdant_upload"
            )

            image_bytes = None
            input_key = None
            valid_image = False

            if uploaded_file is not None:
                image_bytes = uploaded_file.getvalue()

                try:
                    preview = read_image(image_bytes)

                    input_key = (
                        hashlib.sha256(image_bytes).hexdigest(),
                        mode
                    )
                    valid_image = True

                    st.image(preview, width="stretch")
                    st.caption(
                        f"{uploaded_file.name} · "
                        f"{preview.width} × {preview.height} pixels · "
                        f"{len(image_bytes) / 1024:.0f} KB"
                    )

                except Exception as error:
                    st.error(f"Unable to read this image: {error}")

            clicked = st.button(
                "Analyze image →",
                type="primary",
                width="stretch",
                disabled=not valid_image,
                key="single_predict_button"
            )

            st.caption(
                "The first analysis loads the selected model. "
                "Following predictions can be faster."
            )

            if clicked:
                st.session_state.verdant_result = None

                try:
                    with st.spinner("Analyzing image…"):
                        start = time.perf_counter()
                        result = analyze_image(image_bytes, mode)
                        elapsed = time.perf_counter() - start

                    record = {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "filename": uploaded_file.name,
                        "image_sha256": input_key[0],
                        "mode": mode,
                        "model": result["model"],
                        "predicted_label": result["predicted_label"],
                        "model_score": float(result["softmax_score"]),
                        "elapsed_seconds": round(elapsed, 3)
                    }

                    for label, score in result["class_scores"].items():
                        record[f"score_{label}"] = float(score)

                    st.session_state.verdant_result = {
                        "input_key": input_key,
                        "result": result,
                        "record": record
                    }
                    st.session_state.verdant_history.append(record)

                except Exception as error:
                    st.error("The image could not be analyzed.")
                    with st.expander("View error details"):
                        st.code(f"{type(error).__name__}: {error}")

    with right_column:
        with st.container(border=True):
            panel_heading("02", "Analysis results")
            st.caption("Category prediction and model score distribution.")

            current = st.session_state.verdant_result

            if (
                current is not None
                and input_key is not None
                and current["input_key"] == input_key
            ):
                result = current["result"]
                record = current["record"]
                category = LABELS[result["predicted_label"]]

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="eyebrow">Predicted category</div>
                        <h2>{category}</h2>
                        <div class="result-metrics">
                            <div>
                                <strong>{result["softmax_score"]:.2%}</strong>
                                <span>MODEL SCORE</span>
                            </div>
                            <div>
                                <strong>{record["elapsed_seconds"]:.2f}s</strong>
                                <span>ELAPSED TIME</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.caption(
                    f"{result['model']} · "
                    "Time includes preparation and initial model loading."
                )
                render_scores(result["class_scores"])
                render_score_explanation(result["class_scores"])

                st.markdown(
                    """
                    <div class="note">
                        Scores express model preference, not verified diagnostic
                        certainty. A displayed 100% may be rounded.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                render_precautions(
                    result["predicted_label"],
                    aerial=mode == "drone"
                )

                st.download_button(
                    "Download result · JSON",
                    data=json.dumps(record, indent=2),
                    file_name="soybean_prediction.json",
                    mime="application/json",
                    width="stretch",
                    key="single_result_download"
                )

            else:
                st.markdown(
                    """
                    <div class="empty-result">
                        <div class="orbit"><span>❧</span></div>
                        <h3>Ready when you are.</h3>
                        <p>Upload a soybean image and select Analyze image.
                        Your prediction and category scores will appear here.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.caption(
        "The individual-image workspace above runs one selected branch. "
        "The separate fusion workspace below uses its own two inputs."
    )

    st.divider()

    st.markdown(
        """
        <div class="fusion-heading">
            <div class="eyebrow">Separate experimental workspace</div>
            <h2>Leaf–drone fusion</h2>
            <p>Combine ConvNeXt-Tiny and MaxViT-Tiny class scores.
            Upload a leaf photograph and an aerial image here;
            the individual-image upload above is not reused.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.container(border=True):
        panel_heading("03", "Fusion inputs")

        fusion_relationship = st.radio(
            "How are these two images related?",
            [
                "Unverified images — software demonstration",
                "User-confirmed corresponding plot observations"
            ],
            key="fusion_relationship",
            on_change=reset_fusion_inputs
        )

        verified_pair_mode = fusion_relationship.startswith("User-confirmed")

        pair_details = st.text_input(
            "Actual source or pairing details",
            placeholder=(
                "Record image sources or genuine plot/capture information."
            ),
            key="fusion_pair_details",
            on_change=reset_fusion_result
        )

        fusion_leaf_column, fusion_drone_column = st.columns(
            2, gap="large"
        )

        fusion_leaf_bytes = None
        fusion_drone_bytes = None
        fusion_leaf_valid = False
        fusion_drone_valid = False

        with fusion_leaf_column:
            st.markdown("**Smartphone / leaf input**")
            st.caption("Processed by ConvNeXt-Tiny")

            fusion_leaf_file = st.file_uploader(
                "Upload the leaf image",
                type=["jpg", "jpeg", "png", "webp"],
                key="fusion_leaf_upload",
                on_change=reset_fusion_inputs
            )

            if fusion_leaf_file is not None:
                try:
                    fusion_leaf_bytes = fusion_leaf_file.getvalue()
                    leaf_preview = read_image(fusion_leaf_bytes)
                    fusion_leaf_valid = True

                    st.image(
                        leaf_preview,
                        caption=fusion_leaf_file.name,
                        width="stretch"
                    )

                except Exception as error:
                    st.error(f"Leaf image error: {error}")

        with fusion_drone_column:
            st.markdown("**Drone / aerial input**")
            st.caption("Processed by MaxViT-Tiny")

            fusion_drone_file = st.file_uploader(
                "Upload the aerial image",
                type=["jpg", "jpeg", "png", "webp"],
                key="fusion_drone_upload",
                on_change=reset_fusion_inputs
            )

            if fusion_drone_file is not None:
                try:
                    fusion_drone_bytes = fusion_drone_file.getvalue()
                    drone_preview = read_image(fusion_drone_bytes)
                    fusion_drone_valid = True

                    st.image(
                        drone_preview,
                        caption=fusion_drone_file.name,
                        width="stretch"
                    )

                except Exception as error:
                    st.error(f"Drone image error: {error}")

        pairing_confirmed = False

        if verified_pair_mode:
            pairing_confirmed = st.checkbox(
                "I have verified that these images represent the same "
                "plot and relevant capture period, with a common "
                "assessment target.",
                key="fusion_pair_confirmed",
                on_change=reset_fusion_result
            )

            st.caption(
                "The app records your confirmation; it cannot verify "
                "the plot relationship automatically."
            )
        else:
            st.info(
                "Demonstration mode: arbitrary inputs can exercise the "
                "software, but their combined scores are not a valid "
                "assessment of a shared plot."
            )

        st.caption(
            "Fixed baseline: 50% leaf scores + 50% drone scores. "
            "Weights are not optimized; fusion accuracy is not evaluated."
        )

        fusion_input_key = None

        if fusion_leaf_valid and fusion_drone_valid:
            fusion_input_key = (
                hashlib.sha256(fusion_leaf_bytes).hexdigest(),
                hashlib.sha256(fusion_drone_bytes).hexdigest(),
                fusion_relationship,
                pair_details.strip(),
                pairing_confirmed
            )

        fusion_ready = fusion_leaf_valid and fusion_drone_valid

        if verified_pair_mode:
            fusion_ready = (
                fusion_ready
                and pairing_confirmed
                and bool(pair_details.strip())
            )

        fusion_clicked = st.button(
            "Run experimental fusion →",
            type="primary",
            width="stretch",
            disabled=not fusion_ready,
            key="fusion_predict_button"
        )

        if fusion_clicked:
            st.session_state.verdant_fusion_result = None

            try:
                with st.spinner("Analyzing both inputs and combining scores…"):
                    fusion_start = time.perf_counter()

                    fusion_result = analyze_fusion(
                        fusion_leaf_bytes,
                        fusion_drone_bytes
                    )

                    fusion_elapsed = time.perf_counter() - fusion_start

                fusion_record = {
                    **fusion_result,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "leaf_filename": fusion_leaf_file.name,
                    "drone_filename": fusion_drone_file.name,
                    "leaf_sha256": fusion_input_key[0],
                    "drone_sha256": fusion_input_key[1],
                    "input_relationship": fusion_relationship,
                    "source_details": pair_details.strip(),
                    "pairing_confirmed_by_user": pairing_confirmed,
                    "elapsed_seconds": round(fusion_elapsed, 3),
                    "guidance_status": (
                        "Conditional scouting guidance only"
                        if verified_pair_mode
                        else "No crop-level guidance for unverified inputs"
                    )
                }

                st.session_state.verdant_fusion_result = {
                    "input_key": fusion_input_key,
                    "record": fusion_record
                }

                st.session_state.verdant_fusion_history.append(
                    fusion_record
                )

            except Exception as error:
                st.error("Fusion could not be completed.")

                with st.expander("View fusion error details"):
                    st.code(f"{type(error).__name__}: {error}")

    current_fusion = st.session_state.verdant_fusion_result

    if (
        current_fusion is not None
        and fusion_input_key is not None
        and current_fusion["input_key"] == fusion_input_key
    ):
        fusion_record = current_fusion["record"]

        with st.container(border=True):
            panel_heading("04", "Fusion results")

            if not fusion_record["pairing_confirmed_by_user"]:
                st.warning(
                    "UNVERIFIED-INPUT DEMONSTRATION. "
                    "This is a combined score calculation, not a "
                    "validated crop or plot diagnosis."
                )
            else:
                st.info(
                    "Correspondence is user-confirmed. "
                    "This experimental result is not independently validated."
                )

            leaf_metric, drone_metric, combined_metric = st.columns(3)

            leaf_metric.metric(
                "Leaf prediction",
                LABELS[fusion_record["leaf_prediction"]]
            )
            drone_metric.metric(
                "Drone prediction",
                LABELS[fusion_record["drone_prediction"]]
            )
            combined_metric.metric(
                "Combined prediction",
                (
                    LABELS[fusion_record["fused_prediction"]]
                    if fusion_record["fused_prediction"] is not None
                    else "Tied"
                )
            )

            st.caption(
                f"Highest combined score: "
                f"{fusion_record['highest_combined_score']:.2%} · "
                f"Elapsed time: {fusion_record['elapsed_seconds']:.2f}s"
            )

            if fusion_record["branches_agree"]:
                st.success(
                    "Both branches selected the same category. "
                    "Agreement does not prove that the category is correct."
                )
            else:
                st.warning(
                    "The branches disagree. Review the inputs and their "
                    "relationship; averaging does not establish which "
                    "individual prediction is correct."
                )

            score_column, comparison_column = st.columns(
                [1, 1.15], gap="large"
            )

            with score_column:
                st.markdown("**Combined class scores**")
                render_scores(fusion_record["combined_scores"])

            with comparison_column:
                st.markdown("**Branch comparison**")

                fusion_score_table = pd.DataFrame([
                    {
                        "Category": LABELS[label],
                        "Leaf (%)": (
                            fusion_record["leaf_scores"][label] * 100
                        ),
                        "Drone (%)": (
                            fusion_record["drone_scores"][label] * 100
                        ),
                        "Combined (%)": (
                            fusion_record["combined_scores"][label] * 100
                        )
                    }
                    for label in LABELS
                ])

                st.dataframe(
                    fusion_score_table.round(2),
                    hide_index=True,
                    width="stretch"
                )

            render_score_explanation(fusion_record["combined_scores"])

            if fusion_record["pairing_confirmed_by_user"]:
                render_precautions(
                    fusion_record["fused_prediction"],
                    aerial=True,
                    disagreement=not fusion_record["branches_agree"]
                )
            else:
                st.caption(
                    "Crop-level precautions are not inferred from this "
                    "unverified combination. Analyze each image separately "
                    "for conditional inspection guidance."
                )

            st.download_button(
                "Download fusion result · JSON",
                data=json.dumps(fusion_record, indent=2),
                file_name="experimental_fusion_result.json",
                mime="application/json",
                width="stretch",
                key="fusion_result_download"
            )

    with st.expander("Separate fusion history"):
        if st.session_state.verdant_fusion_history:
            fusion_history_rows = []

            for saved_record in st.session_state.verdant_fusion_history:
                history_row = {
                    "timestamp_utc": saved_record["timestamp_utc"],
                    "leaf_filename": saved_record["leaf_filename"],
                    "drone_filename": saved_record["drone_filename"],
                    "input_relationship": saved_record["input_relationship"],
                    "source_details": saved_record["source_details"],
                    "leaf_prediction": saved_record["leaf_prediction"],
                    "drone_prediction": saved_record["drone_prediction"],
                    "fused_prediction": (
                        saved_record["fused_prediction"] or "tied"
                    ),
                    "highest_combined_score": (
                        saved_record["highest_combined_score"]
                    ),
                    "elapsed_seconds": saved_record["elapsed_seconds"]
                }

                for label in LABELS:
                    history_row[f"leaf_score_{label}"] = (
                        saved_record["leaf_scores"][label]
                    )
                    history_row[f"drone_score_{label}"] = (
                        saved_record["drone_scores"][label]
                    )
                    history_row[f"combined_score_{label}"] = (
                        saved_record["combined_scores"][label]
                    )

                fusion_history_rows.append(history_row)

            fusion_history_frame = pd.DataFrame(fusion_history_rows)

            st.dataframe(
                fusion_history_frame[[
                    "leaf_filename",
                    "drone_filename",
                    "input_relationship",
                    "fused_prediction",
                    "highest_combined_score"
                ]],
                hide_index=True,
                width="stretch"
            )

            st.download_button(
                "Export fusion history · CSV",
                data=fusion_history_frame.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="experimental_fusion_history.csv",
                mime="text/csv",
                key="fusion_history_download"
            )

            if st.button(
                "Clear fusion history",
                key="clear_fusion_history"
            ):
                st.session_state.verdant_fusion_history = []
                st.rerun()
        else:
            st.caption("No fusion runs have been recorded in this session.")

        st.caption(
            "Fusion history is separate from individual-image history. "
            "Download it before closing or restarting the app."
        )

with history_tab:
    st.subheader("Your analysis trail")
    st.caption(
        "Review and export individual-image predictions from this browser "
        "session. This log does not measure prediction accuracy."
    )

    if st.session_state.verdant_history:
        history = pd.DataFrame(st.session_state.verdant_history)

        metric_left, metric_right = st.columns(2)
        metric_left.metric("Predictions recorded", len(history))
        metric_right.metric(
            "Unique image files",
            history["image_sha256"].nunique()
        )

        history_display = history[[
            "filename", "mode", "model",
            "predicted_label", "model_score", "elapsed_seconds"
        ]].copy()

        history_display.columns = [
            "Image", "Source", "Model",
            "Prediction", "Model score", "Elapsed seconds"
        ]
        history_display["Prediction"] = history_display[
            "Prediction"
        ].map(LABELS)
        history_display["Model score"] = history_display[
            "Model score"
        ].map(lambda value: f"{value:.2%}")

        st.dataframe(
            history_display,
            hide_index=True,
            width="stretch"
        )

        download_column, clear_column = st.columns([3, 1])

        with download_column:
            st.download_button(
                "Export session · CSV",
                data=history.to_csv(index=False).encode("utf-8"),
                file_name="soybean_prediction_history.csv",
                mime="text/csv",
                key="single_history_download"
            )

        with clear_column:
            if st.button(
                "Clear history",
                width="stretch",
                key="clear_single_history"
            ):
                st.session_state.verdant_history = []
                st.rerun()

        st.caption("Download the history before closing or restarting the app.")

    else:
        st.markdown(
            """
            <div class="empty-result">
                <div class="orbit"><span>≡</span></div>
                <h3>A fresh workspace.</h3>
                <p>Your predictions will appear here after you analyze an image.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

with evidence_tab:
    st.subheader("The models behind the predictions")
    st.caption(
        "The selected models were chosen using validation macro-F1 "
        "before their V2 test evaluation."
    )

    evidence = pd.DataFrame([
        {
            "Branch": "Leaf",
            "Model": "ConvNeXt-Tiny",
            "Saved epoch": 14,
            "Test images": 316,
            "Test accuracy": "81.65%",
            "Test macro-F1": "0.8422"
        },
        {
            "Branch": "Drone",
            "Model": "MaxViT-Tiny",
            "Saved epoch": 6,
            "Test images": 440,
            "Test accuracy": "100.00%",
            "Test macro-F1": "1.0000"
        }
    ])

    st.dataframe(evidence, hide_index=True, width="stretch")

    st.markdown(
        """
        <div class="note">
            The drone result is exploratory. It describes this specific test
            split and does not establish 100% accuracy on new farms or images.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("How an image becomes a prediction", expanded=True):
        st.markdown(
            """
            1. Select the **leaf** or **drone** branch.
            2. The image is oriented, converted to RGB, and resized with padding.
            3. The saved model produces scores for the four soybean categories.
            4. The highest-scoring category is displayed.

            Leaf input size: **384 × 384**.  
            Drone input size: **224 × 224**.
            """
        )

    with st.expander("How experimental fusion works"):
        st.markdown(
            """
            The separate fusion workspace runs both saved models and computes:

            **Combined score = 0.5 × leaf score + 0.5 × drone score**

            The category with the highest combined score is displayed.
            Near-equal top scores are reported as a tie.

            - The weights are a fixed starting baseline.
            - No joint training is performed by this operation.
            - The app cannot independently verify input correspondence.
            - Unverified inputs are labeled as software demonstrations.
            - Fusion accuracy and improvement have not been evaluated.
            - Individual test accuracies are not fusion accuracy.
            """
        )

    with st.expander("Evaluation boundaries"):
        st.markdown(
            """
            - Different drone recordings can share fields or capture dates.
            - Healthy drone image source independence is unverified.
            - Healthy and diseased leaf images came from different years.
            - External generalization and early-stage detection are not established.
            - Other crops and unrelated images are not automatically rejected.
            - Verified matching leaf–drone observations with reliable target
              labels are needed for fusion evaluation.
            - Model percentages do not measure severity or infected area.
            - Precautions are general inspection guidance, not treatment prescriptions.
            """
        )

st.markdown(
    """
    <div class="footer">
        <span>VERDANT / SOYBEAN HEALTH ASSESSMENT</span>
        <span>Research demonstration · Verify predictions before field decisions</span>
    </div>
    """,
    unsafe_allow_html=True
)