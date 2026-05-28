"""
Standalone Streamlit app for surface-roughness classification.

Loads the trained model directly via the project's Predictor — no FastAPI
backend required — so the whole demo can run as a single Streamlit Community
Cloud deployment.

Checkpoint location is resolved in this order:
  1. st.secrets["checkpoint_path"]      (recommended on Streamlit Cloud)
  2. CHECKPOINT_PATH environment variable
  3. ./outputs/demo_model.pth            (local default)

The value may be a local path **or** an http(s) URL; URLs are downloaded once
into the temp directory (a 94 MB checkpoint doesn't fit in Streamlit secrets,
so URL-based hosting is the practical path for a public deploy).
"""
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.predictor import Predictor  # noqa: E402

GRIT_VALUES = [60, 80, 100, 120, 150, 180, 240, 320, 400, 600]
DEFAULT_GRIT = 150
LABEL_COLOURS = {"Smooth": "#2ecc71", "Medium": "#f39c12", "Rough": "#e74c3c"}

SAMPLES_DIR = PROJECT_ROOT / "data" / "sample_images"
SAMPLES = [
    {"file": "smooth_ra_0.45.png", "cls": "Smooth", "ra": 0.45, "grit": 400},
    {"file": "smooth_ra_0.52.png", "cls": "Smooth", "ra": 0.52, "grit": 320},
    {"file": "medium_ra_0.78.png", "cls": "Medium", "ra": 0.78, "grit": 180},
    {"file": "medium_ra_0.84.png", "cls": "Medium", "ra": 0.84, "grit": 150},
    {"file": "rough_ra_1.12.png",  "cls": "Rough",  "ra": 1.12, "grit": 100},
    {"file": "rough_ra_1.55.png",  "cls": "Rough",  "ra": 1.55, "grit":  60},
]

REPO_URL = "https://github.com/NassabOussama/surface-roughness-ml"


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def _resolve_checkpoint():
    raw = None
    try:
        raw = str(st.secrets["checkpoint_path"])
    except Exception:
        pass
    if not raw:
        raw = os.environ.get("CHECKPOINT_PATH")
    if not raw:
        raw = str(PROJECT_ROOT / "outputs" / "demo_model.pth")

    if raw.startswith(("http://", "https://")):
        cache = Path(tempfile.gettempdir()) / "surface-roughness-ml-cache"
        cache.mkdir(parents=True, exist_ok=True)
        local = cache / f"model_{hashlib.md5(raw.encode()).hexdigest()[:10]}.pth"
        if not local.exists():
            with st.spinner("Downloading model checkpoint (one-time)…"):
                urllib.request.urlretrieve(raw, local)
        return str(local)
    return raw


@st.cache_resource(show_spinner="Loading model…")
def _load_predictor(checkpoint_path):
    return Predictor(checkpoint_path)


# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------

def _select_sample(sample):
    st.session_state["image"] = Image.open(SAMPLES_DIR / sample["file"]).convert("RGB")
    st.session_state["source_name"] = f"sample: {sample['file']}"
    st.session_state["grit"] = sample["grit"]
    st.session_state["result"] = None


def _on_upload():
    f = st.session_state.get("_uploader")
    if f is not None:
        st.session_state["image"] = Image.open(f).convert("RGB")
        st.session_state["source_name"] = f"upload: {f.name}"
        st.session_state["result"] = None


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

st.set_page_config(page_title="Surface Roughness Classifier", layout="centered")
st.title("Surface Roughness Classifier")
st.caption(
    "Predict the roughness class of a machined metal surface — Smooth, Medium, "
    "or Rough — from a microscopy image and the abrasive grit value."
)
st.markdown(
    f"📖 **About this demo** — A portfolio project by **Oussama Nassab** (UQAR). "
    f"The model is a **FiLM-ResNet50** conditioned on the grit value. "
    f"Code, training pipeline, and rigorous benchmark: "
    f"**[{REPO_URL.removeprefix('https://')}]({REPO_URL})**."
)

if "grit" not in st.session_state:
    st.session_state["grit"] = DEFAULT_GRIT

# Resolve + load the model (cached for the lifetime of the Streamlit process)
try:
    ckpt = _resolve_checkpoint()
    if not Path(ckpt).exists():
        st.error(
            f"❌ Model checkpoint not found at `{ckpt}`. "
            "Set `checkpoint_path` in Streamlit secrets, the `CHECKPOINT_PATH` "
            "environment variable, or place the .pth at the default location. "
            "Either a local path or an http(s) URL is accepted."
        )
        st.stop()
    predictor = _load_predictor(ckpt)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.selectbox(
        "Grit value used during machining",
        GRIT_VALUES,
        key="grit",
        help="Grit grade of the abrasive used. The same visual texture implies "
        "different roughness depending on the abrasive — the model needs this input.",
    )
    st.markdown("---")
    st.caption(f"Device: `{predictor.device}`")

# Sample picker
st.subheader("Try a sample image")
st.caption("Click any sample to load it — the grit value updates automatically.")

cols = st.columns(3)
for i, s in enumerate(SAMPLES):
    with cols[i % 3]:
        st.image(str(SAMPLES_DIR / s["file"]), use_container_width=True)
        st.button(
            f"{s['cls']} · Ra {s['ra']} µm · grit {s['grit']}",
            key=f"btn_{s['file']}",
            on_click=_select_sample,
            args=(s,),
            use_container_width=True,
        )

st.divider()

# Upload
st.subheader("…or upload your own")
st.file_uploader(
    "Microscopy image (PNG or JPEG)",
    type=["png", "jpg", "jpeg"],
    key="_uploader",
    on_change=_on_upload,
)

# Current input + classify
image = st.session_state.get("image")
if image is not None:
    st.divider()
    st.markdown(
        f"**Loaded:** `{st.session_state.get('source_name', '')}` "
        f"· **grit** `{st.session_state['grit']}`"
    )
    st.image(image, caption="Input image", use_container_width=True)

    if st.button("Classify", type="primary", use_container_width=True):
        with st.spinner("Running inference…"):
            try:
                st.session_state["result"] = predictor.predict(
                    image, st.session_state["grit"]
                )
            except ValueError as e:
                st.session_state["result"] = None
                st.error(f"Prediction error: {e}")

# Result
result = st.session_state.get("result")
if result is not None:
    label = result["label"]
    conf = result["confidence"]
    probs = result["probabilities"]
    colour = LABEL_COLOURS.get(label, "#999")
    st.markdown(
        f"<h2 style='color:{colour}'>Prediction: {label}</h2>",
        unsafe_allow_html=True,
    )
    st.metric("Confidence", f"{conf:.1%}")
    st.subheader("Class probabilities")
    for cls, prob in probs.items():
        st.markdown(f"**{cls}**")
        st.progress(prob, text=f"{prob:.1%}")
