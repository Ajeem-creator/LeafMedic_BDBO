import os
import sys
import json
import time
import zipfile
import tempfile
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
import streamlit as st
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LeafMedic AI – Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Recommended runtime: Python 3.11, TensorFlow 2.16.1, Keras 3.3.3.
if sys.version_info < (3, 11):
    raise RuntimeError(
        f"Python 3.11 or newer is required. Current version: {sys.version.split()[0]}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Premium CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #042f1e 0%, #064e3b 100%);
}
[data-testid="stSidebar"] * { color: #d1fae5 !important; }
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    color: #d1fae5 !important;
    border-radius: 10px;
    padding: 10px 14px;
    font-weight: 600;
    text-align: left;
    transition: all 0.2s ease;
    margin-bottom: 4px;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(52,211,153,0.2);
    border-color: #34d399;
}

/* Hero Banner */
.hero {
    background: linear-gradient(135deg, #064e3b 0%, #047857 50%, #059669 100%);
    padding: 28px 32px;
    border-radius: 18px;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 12px 28px -6px rgba(6,78,59,0.35);
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    top: -50px; right: -50px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(52,211,153,0.2) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 { font-size: 2rem; font-weight: 800; margin-bottom: 6px; }
.hero p  { color: #a7f3d0; font-size: 0.95rem; max-width: 800px; }

/* Stat pills */
.pill-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }
.pill {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(6px);
    padding: 5px 14px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.2);
}

/* Cards */
.card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.04);
    margin-bottom: 18px;
}
.card-accent {
    border-left: 4px solid #10b981;
}

/* Result banner */
.result-healthy {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 2px solid #10b981;
    border-radius: 14px;
    padding: 20px 24px;
}
.result-disease {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border: 2px solid #ef4444;
    border-radius: 14px;
    padding: 20px 24px;
}

/* Metric boxes */
.m-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin: 18px 0; }
.m-box {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 18px 20px; text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}
.m-box .label { font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: #64748b; }
.m-box .value { font-size: 1.7rem; font-weight: 800; color: #0f172a; margin-top: 2px; }
.m-box .sub   { font-size: 0.78rem; color: #64748b; margin-top: 2px; }

/* Treatment grid */
.treat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin-top: 16px; }
.treat-card {
    background: #f8fafc; border-left: 4px solid #10b981; border-radius: 0 12px 12px 0;
    padding: 16px 18px;
}
.treat-card h4 { margin: 0 0 6px 0; font-size: 0.92rem; color: #0f172a; }
.treat-card p  { margin: 0; font-size: 0.85rem; color: #475569; line-height: 1.45; }

/* Tip boxes */
.tip-item {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 6px;
    font-size: 0.88rem;
    color: #166534;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading (with Keras 3 compatibility fix)
# ─────────────────────────────────────────────────────────────────────────────
def _patch_keras_layer_from_config():
    """Ensure Keras 3.12+ does not fail on quantization_config: None from saved configs."""
    try:
        import keras
        orig_from_config = keras.layers.Layer.from_config

        @classmethod
        def _safe_from_config(cls, config):
            if isinstance(config, dict):
                config.pop("quantization_config", None)
            return orig_from_config.__func__(cls, config)

        keras.layers.Layer.from_config = _safe_from_config
    except Exception:
        pass


_patch_keras_layer_from_config()


def _fix_keras_archive(src, dst):
    """Strip quantization_config from .keras zip to fix Keras 3.12+ compat."""
    if os.path.exists(dst):
        return dst
    if not os.path.exists(src):
        return dst
    try:
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(src, "r") as z:
                z.extractall(tmp)
            cfg = os.path.join(tmp, "config.json")
            if os.path.exists(cfg):
                with open(cfg, "r") as f:
                    data = json.load(f)

                def _strip(obj):
                    if isinstance(obj, dict):
                        obj.pop("quantization_config", None)
                        for v in obj.values():
                            _strip(v)
                    elif isinstance(obj, list):
                        for v in obj:
                            _strip(v)

                _strip(data)
                with open(cfg, "w") as f:
                    json.dump(data, f)
            with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
                for root, _, files in os.walk(tmp):
                    for fn in files:
                        fp = os.path.join(root, fn)
                        z.write(fp, os.path.relpath(fp, tmp))
        return dst
    except Exception:
        return src


@st.cache_resource(show_spinner="🌿 Loading CNN & BDBO models …")
def load_all():
    import sys
    import keras

    # Check for Keras 3+ requirement
    try:
        k_major = int(keras.__version__.split(".")[0])
        if k_major < 3:
            raise RuntimeError(
                f"Keras version {keras.__version__} is incompatible (Keras 3.0+ required).\n\n"
                f"The models were trained and saved in Keras 3 (.keras) format, which Keras 2 cannot open.\n\n"
                f"Currently using Python at:\n{sys.executable}\n\n"
                f"SOLUTION: Use Python 3.11 with TensorFlow 2.16.1 and Keras 3.3.3. "
                f"Activate the Python 3.11 virtual environment, then run: python -m streamlit run app.py"
            )
    except (ValueError, IndexError):
        pass

    _patch_keras_layer_from_config()

    cnn_compat = os.path.join(BASE_DIR, "best_leaf_disease_model_compat.keras")
    cnn_orig = os.path.join(BASE_DIR, "best_leaf_disease_model.keras")
    cnn_path = cnn_compat if os.path.exists(cnn_compat) else _fix_keras_archive(cnn_orig, cnn_compat)
    if not os.path.exists(cnn_path):
        cnn_path = cnn_orig

    dbo_compat = os.path.join(BASE_DIR, "CNN_DBO_final_classifier_compat.keras")
    dbo_orig = os.path.join(BASE_DIR, "CNN_DBO_final_classifier.keras")
    dbo_path = dbo_compat if os.path.exists(dbo_compat) else _fix_keras_archive(dbo_orig, dbo_compat)
    if not os.path.exists(dbo_path):
        dbo_path = dbo_orig

    cnn_model = keras.models.load_model(cnn_path, compile=False)
    dbo_model = keras.models.load_model(dbo_path, compile=False)

    # Build feature extractor (256‑D GAP layer)
    feat_layer = None
    for layer in cnn_model.layers:
        try:
            s = layer.output.shape
            if len(s) == 2 and s[-1] == 256:
                feat_layer = layer
                break
        except (AttributeError, TypeError):
            continue
    if feat_layer is None:
        raise RuntimeError(
            "Could not find the 256-D GAP feature layer in the CNN model. "
            "Check that the correct model file is used."
        )
    feat_ext = keras.models.Model(inputs=cnn_model.input, outputs=feat_layer.output)

    with open(os.path.join(BASE_DIR, "class_names.json"), "r", encoding="utf-8") as f:
        class_names = json.load(f)

    dbo_indices = np.asarray(
        np.load(os.path.join(BASE_DIR, "DBO_feature_indices.npy"))
    ).reshape(-1).astype(np.int64)
    dbo_mask = np.asarray(
        np.load(os.path.join(BASE_DIR, "DBO_feature_mask.npy"))
    ).reshape(-1)

    hist_path = os.path.join(BASE_DIR, "training_hist (1).json")
    hist = {}
    if os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            hist = json.load(f)

    return cnn_model, dbo_model, feat_ext, class_names, dbo_indices, dbo_mask, hist


LOAD_ERR = ""
try:
    CNN, DBO, FEAT_EXT, CLASS_NAMES, DBO_IDX, DBO_MASK, HIST = load_all()
    OK = True
except Exception as e:
    import traceback
    OK = False
    LOAD_ERR = f"{e}\n\n{traceback.format_exc()}"
    # Provide safe defaults so pages that reference these don't crash
    CLASS_NAMES = []
    DBO_IDX = np.array([], dtype=np.int64)
    DBO_MASK = np.array([], dtype=np.int64)
    HIST = {}


# ─────────────────────────────────────────────────────────────────────────────
# Disease Knowledge Base (38 classes)
# ─────────────────────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "crop": "Apple", "disease": "Apple Scab",
        "pathogen": "Venturia inaequalis (Fungus)",
        "symptoms": "Olive-green to black velvety spots on leaves and fruit; leaves curl and drop early.",
        "tips": [
            "Apply sulfur or copper fungicides at bud break.",
            "Prune trees to improve air circulation.",
            "Rake and destroy fallen leaves every autumn.",
            "Plant resistant cultivars (Liberty, Enterprise, Freedom).",
        ],
    },
    "Apple___Black_rot": {
        "crop": "Apple", "disease": "Black Rot",
        "pathogen": "Botryosphaeria obtusa (Fungus)",
        "symptoms": "Circular brown leaf spots with purple margins; mummified black fruit on the tree.",
        "tips": [
            "Prune dead wood and cankers 15 cm below infected tissue.",
            "Remove mummified fruit from trees and ground.",
            "Apply Captan or Thiophanate-methyl from silver tip to petal fall.",
            "Disinfect pruning shears with 70 % alcohol between cuts.",
        ],
    },
    "Apple___Cedar_apple_rust": {
        "crop": "Apple", "disease": "Cedar Apple Rust",
        "pathogen": "Gymnosporangium juniperi-virginianae",
        "symptoms": "Bright yellow-orange spots on upper leaf surfaces with tiny black dots; tube-like structures below.",
        "tips": [
            "Remove nearby eastern red cedar / juniper alternate hosts.",
            "Apply Myclobutanil or Mancozeb when cedar galls open in spring.",
            "Plant rust-immune varieties (Enterprise, Redfree, William's Pride).",
        ],
    },
    "Apple___healthy": {
        "crop": "Apple", "disease": "Healthy",
        "pathogen": "None",
        "symptoms": "Vibrant green foliage, no lesions or discoloration.",
        "tips": [
            "Maintain proper watering & balanced fertilization.",
            "Prune annually for airflow.",
            "Monitor for pests regularly.",
        ],
    },
    "Blueberry___healthy": {
        "crop": "Blueberry", "disease": "Healthy",
        "pathogen": "None",
        "symptoms": "Clean, waxy leaves with healthy deep-green pigmentation.",
        "tips": [
            "Keep soil pH 4.5–5.5 with sulfur amendments.",
            "Use pine bark mulch for acidity.",
            "Drip-irrigate at the base to keep foliage dry.",
        ],
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "crop": "Cherry", "disease": "Powdery Mildew",
        "pathogen": "Podosphaera clandestina (Fungus)",
        "symptoms": "White powdery growth on leaves and shoots; leaves curl upward.",
        "tips": [
            "Spray potassium bicarbonate or wettable sulfur early in the season.",
            "Prune inner water sprouts to reduce humidity.",
            "Avoid excess nitrogen fertilization.",
            "Rotate fungicides (Quinoxyfen, Myclobutanil) to prevent resistance.",
        ],
    },
    "Cherry_(including_sour)___healthy": {
        "crop": "Cherry", "disease": "Healthy",
        "pathogen": "None",
        "symptoms": "Uniform green leaves with no spotting or gummosis.",
        "tips": [
            "Prune during dry summer to reduce canker risk.",
            "Ensure good drainage and pollination conditions.",
        ],
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "crop": "Corn", "disease": "Gray Leaf Spot (Cercospora)",
        "pathogen": "Cercospora zeae-maydis",
        "symptoms": "Long narrow rectangular tan-to-gray lesions restricted by leaf veins.",
        "tips": [
            "Rotate crops for 2–3 years away from corn.",
            "Plant hybrids with GLS tolerance ratings.",
            "Apply Azoxystrobin + Propiconazole at tasseling (VT–R1).",
            "Deep-plow previous crop debris to reduce inoculum.",
        ],
    },
    "Corn_(maize)___Common_rust_": {
        "crop": "Corn", "disease": "Common Rust",
        "pathogen": "Puccinia sorghi (Fungus)",
        "symptoms": "Small cinnamon-brown pustules on both leaf surfaces.",
        "tips": [
            "Use rust-resistant corn hybrids.",
            "Apply Pyraclostrobin or Tebuconazole if pustules appear before silking.",
            "Plant early in the season to avoid peak spore season.",
        ],
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "crop": "Corn", "disease": "Northern Leaf Blight",
        "pathogen": "Exserohilum turcicum",
        "symptoms": "Cigar-shaped elongated grayish-green to tan lesions 2.5–15 cm long.",
        "tips": [
            "Plant hybrids with Ht1, Ht2, Ht3, or HtN resistance genes.",
            "Rotate with non-host crops (soybean, small grains).",
            "Apply strobilurin + triazole premix at disease onset.",
        ],
    },
    "Corn_(maize)___healthy": {
        "crop": "Corn", "disease": "Healthy",
        "pathogen": "None",
        "symptoms": "Broad deep-green leaves with vigorous stalk development.",
        "tips": [
            "Maintain balanced N-P-K fertilization.",
            "Scout fields bi-weekly from V6 through R3.",
        ],
    },
    "Grape___Black_rot": {
        "crop": "Grape", "disease": "Black Rot",
        "pathogen": "Guignardia bidwellii (Fungus)",
        "symptoms": "Reddish-brown circular leaf spots; shriveled mummified black berries.",
        "tips": [
            "Remove all mummies from trellis and vineyard floor.",
            "Apply Mancozeb or Tebuconazole from early shoot growth through fruit set.",
            "Canopy management – leaf pulling & shoot positioning for rapid drying.",
        ],
    },
    "Grape___Esca_(Black_Measles)": {
        "crop": "Grape", "disease": "Esca (Black Measles)",
        "pathogen": "Phaeoacremonium & Fomitiporia spp.",
        "symptoms": "'Tiger-stripe' interveinal chlorosis and necrosis; sudden vine wilting.",
        "tips": [
            "Excise infected cordons; seal pruning cuts with wound paste.",
            "Do not prune during rainy weather.",
            "Apply Trichoderma biocontrols to pruning wounds.",
        ],
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "crop": "Grape", "disease": "Leaf Blight (Isariopsis)",
        "pathogen": "Phaeoisariopsis vitis",
        "symptoms": "Large irregular dark-brown necrotic lesions; premature leaf drop.",
        "tips": [
            "Apply copper oxychloride or Bordeaux mixture before wet periods.",
            "Ensure good vine spacing; clear fallen debris.",
        ],
    },
    "Grape___healthy": {
        "crop": "Grape", "disease": "Healthy",
        "pathogen": "None",
        "symptoms": "Healthy palmate leaves with firm cluster stems.",
        "tips": [
            "Maintain trellising and drip irrigation.",
            "Monitor for phylloxera and leafhoppers.",
        ],
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "crop": "Orange", "disease": "Huanglongbing (Citrus Greening)",
        "pathogen": "Candidatus Liberibacter asiaticus (Psyllid-vectored)",
        "symptoms": "Asymmetric blotchy mottle crossing veins; small lopsided bitter fruit.",
        "tips": [
            "Control psyllid vectors with systemic insecticides (Imidacloprid).",
            "Remove and destroy infected trees promptly.",
            "Plant only certified disease-free nursery stock.",
            "Release Tamarixia radiata parasitoid wasps for biological control.",
        ],
    },
    "Peach___Bacterial_spot": {
        "crop": "Peach", "disease": "Bacterial Spot",
        "pathogen": "Xanthomonas arboricola pv. pruni",
        "symptoms": "Small purple-black polygonal lesions; centers drop out giving 'shot-hole' look.",
        "tips": [
            "Plant resistant varieties (Candor, Dixired).",
            "Apply copper bactericides during dormant season.",
            "Avoid overhead irrigation.",
        ],
    },
    "Peach___healthy": {
        "crop": "Peach", "disease": "Healthy",
        "pathogen": "None",
        "symptoms": "Smooth lanceolate vibrant-green leaves, no curl or gummosis.",
        "tips": [
            "Prune to open-center canopy for sunlight penetration.",
            "Apply dormant oil spray for scale prevention.",
        ],
    },
    "Pepper,_bell___Bacterial_spot": {
        "crop": "Bell Pepper", "disease": "Bacterial Spot",
        "pathogen": "Xanthomonas euvesicatoria",
        "symptoms": "Small water-soaked spots turning brown with yellow halos; severe defoliation.",
        "tips": [
            "Use disease-free seeds and transplants.",
            "Avoid overhead watering; use drip lines.",
            "Apply copper hydroxide + Mancozeb tank mix every 7–10 days.",
            "Rotate with non-solanaceous crops for 2–3 years.",
        ],
    },
    "Pepper,_bell___healthy": {
        "crop": "Bell Pepper", "disease": "Healthy",
        "pathogen": "None",
        "symptoms": "Dark green glossy leaves with firm petioles.",
        "tips": [
            "Consistent drip irrigation to prevent blossom end rot.",
            "Use plastic mulch for weed and moisture control.",
        ],
    },
    "Potato___Early_blight": {
        "crop": "Potato", "disease": "Early Blight",
        "pathogen": "Alternaria solani (Fungus)",
        "symptoms": "Concentric 'target-board' ring lesions on older leaves.",
        "tips": [
            "Use certified disease-free seed tubers.",
            "Rotate crops annually.",
            "Apply Chlorothalonil or Mancozeb at first symptoms.",
            "Maintain proper nitrogen levels.",
        ],
    },
    "Potato___Late_blight": {
        "crop": "Potato", "disease": "Late Blight",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "symptoms": "Rapidly expanding dark water-soaked lesions; white cottony growth on undersides.",
        "tips": [
            "Plant blight-resistant varieties.",
            "Apply Mefenoxam or Cyazofamid preventatively before wet weather.",
            "Destroy volunteer potato plants and cull piles.",
            "Avoid overhead watering; improve field drainage.",
        ],
    },
    "Potato___healthy": {
        "crop": "Potato", "disease": "Healthy",
        "pathogen": "None",
        "symptoms": "Robust bushy haulm with deep-green pinnate leaves.",
        "tips": [
            "Hill rows properly to prevent tuber greening.",
            "Ensure balanced potassium and phosphorus nutrition.",
        ],
    },
    "Raspberry___healthy": {
        "crop": "Raspberry", "disease": "Healthy",
        "pathogen": "None",
        "symptoms": "Vigorous canes with vibrant serrated green leaves.",
        "tips": [
            "Prune out spent floricanes after fruiting.",
            "Trellis canes and use wood-chip mulch.",
        ],
    },
    "Soybean___healthy": {
        "crop": "Soybean", "disease": "Healthy",
        "pathogen": "None",
        "symptoms": "Trifoliate leaves with uniform chlorophyll and vigorous nodulation.",
        "tips": [
            "Inoculate seeds with Bradyrhizobium japonicum.",
            "Scout weekly for soybean rust and sudden death syndrome.",
        ],
    },
    "Squash___Powdery_mildew": {
        "crop": "Squash", "disease": "Powdery Mildew",
        "pathogen": "Podosphaera xanthii (Fungus)",
        "symptoms": "White powdery fungal colonies on leaves, petioles and stems.",
        "tips": [
            "Plant PM-resistant squash hybrids.",
            "Apply diluted milk spray (10–20 %) or neem oil in morning.",
            "Ensure wide row spacing for solar penetration.",
            "Apply Triflumizole or Myclobutanil at first white spots.",
        ],
    },
    "Strawberry___Leaf_scorch": {
        "crop": "Strawberry", "disease": "Leaf Scorch",
        "pathogen": "Diplocarpon earlianum (Fungus)",
        "symptoms": "Dark purple-to-black irregular spots without white centers; foliage looks scorched.",
        "tips": [
            "Remove and burn heavily infected leaves.",
            "Apply Captan or Pyraclostrobin before flowering.",
            "Plant in raised beds for rapid water drainage.",
            "Renovate beds annually.",
        ],
    },
    "Strawberry___healthy": {
        "crop": "Strawberry", "disease": "Healthy",
        "pathogen": "None",
        "symptoms": "Low rosette of vibrant green trifoliate leaves with sturdy runners.",
        "tips": [
            "Use straw mulch under fruit to prevent soil contact.",
            "Rotate beds every 3–4 years.",
        ],
    },
    "Tomato___Bacterial_spot": {
        "crop": "Tomato", "disease": "Bacterial Spot",
        "pathogen": "Xanthomonas perforans",
        "symptoms": "Small dark-brown water-soaked lesions (<3 mm); leaves yellow and drop.",
        "tips": [
            "Use disease-free seeds; sanitize with bleach or hot water.",
            "Spray copper + Mancozeb or Actigard plant activator.",
            "Avoid working in fields when foliage is wet.",
        ],
    },
    "Tomato___Early_blight": {
        "crop": "Tomato", "disease": "Early Blight",
        "pathogen": "Alternaria linariae (Fungus)",
        "symptoms": "Concentric 'bullseye' brown lesions with yellow chlorotic halo on lower leaves.",
        "tips": [
            "Prune lower 30 cm of foliage touching soil.",
            "Apply Chlorothalonil or Mancozeb at 7–14 day intervals.",
            "Mulch soil surface with plastic/straw to prevent rain splash.",
            "3-year solanaceous crop rotation.",
        ],
    },
    "Tomato___Late_blight": {
        "crop": "Tomato", "disease": "Late Blight",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "symptoms": "Pale green to oily brown water-soaked lesions; white sporangial down below in wet air.",
        "tips": [
            "Apply copper fungicide immediately on detection.",
            "Use resistant cultivars (Defiant, Iron Lady).",
            "Destroy heavily infected plants in sealed bags.",
            "Spray Mandipropamid or Dimethomorph preventatively.",
        ],
    },
    "Tomato___Leaf_Mold": {
        "crop": "Tomato", "disease": "Leaf Mold",
        "pathogen": "Passalora fulva (Fungus)",
        "symptoms": "Pale greenish-yellow spots above; velvety olive-green mold beneath leaves.",
        "tips": [
            "Increase greenhouse ventilation; lower humidity < 85 %.",
            "Use varieties with Cf-resistance genes.",
            "Apply Boscalid or Chlorothalonil when spots first appear.",
        ],
    },
    "Tomato___Septoria_leaf_spot": {
        "crop": "Tomato", "disease": "Septoria Leaf Spot",
        "pathogen": "Septoria lycopersici (Fungus)",
        "symptoms": "Numerous 1–3 mm spots with dark margins and gray-white centers dotted with black pycnidia.",
        "tips": [
            "Strip lower leaves as plant grows.",
            "Apply Chlorothalonil or Azoxystrobin when first fruit clusters set.",
            "Drip irrigate to keep foliage dry.",
            "Clean all stakes and cages at end of season.",
        ],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "crop": "Tomato", "disease": "Spider Mite Infestation",
        "pathogen": "Tetranychus urticae (Arachnid Pest)",
        "symptoms": "Fine yellow stippling on leaf surface; silken webbing underneath; leaves bronze.",
        "tips": [
            "Release predatory mites (Phytoseiulus persimilis).",
            "Spray insecticidal soap or neem oil.",
            "Keep plants well-watered – mites thrive in hot dry conditions.",
            "Rotate Abamectin, Bifenazate to prevent resistance.",
        ],
    },
    "Tomato___Target_Spot": {
        "crop": "Tomato", "disease": "Target Spot",
        "pathogen": "Corynespora cassiicola (Fungus)",
        "symptoms": "Small brown circular lesions with concentric rings; can affect fruit too.",
        "tips": [
            "Ensure wide plant spacing for airy canopy.",
            "Apply Pyraclostrobin + Boscalid during humid weather.",
            "Prune excess suckers; eliminate weed hosts.",
        ],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "crop": "Tomato", "disease": "Yellow Leaf Curl Virus (TYLCV)",
        "pathogen": "Begomovirus (Whitefly-vectored)",
        "symptoms": "Severe stunting, upward leaf cupping, interveinal chlorosis, flower abortion.",
        "tips": [
            "Control whitefly vectors with Dinotefuran or Spirotetramat.",
            "Use yellow sticky traps and silver reflective mulches.",
            "Plant TYLCV-resistant hybrids with Ty-1/Ty-2/Ty-3 genes.",
            "Use 50-mesh insect screens in nurseries.",
        ],
    },
    "Tomato___Tomato_mosaic_virus": {
        "crop": "Tomato", "disease": "Tomato Mosaic Virus (ToMV)",
        "pathogen": "Tobamovirus (Mechanically Transmitted)",
        "symptoms": "Light-and-dark green mosaic mottling, leaf distortion, internal fruit browning.",
        "tips": [
            "Sterilize tools with 20 % milk or 10 % TSP solution.",
            "Wash hands with soap before touching plants.",
            "Eradicate infected plants immediately – no chemical cure exists.",
            "Smokers must wear gloves when handling seedlings.",
        ],
    },
    "Tomato___healthy": {
        "crop": "Tomato", "disease": "Healthy",
        "pathogen": "None",
        "symptoms": "Deep green compound leaves with robust stem and healthy blossoms.",
        "tips": [
            "Consistent watering schedule to prevent blossom end rot.",
            "Stake or cage plants early.",
            "Scout leaf undersides weekly.",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────
def _parse(label):
    parts = label.split("___")
    crop = parts[0].replace("_", " ").replace(",", ", ").strip()
    dis = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    return crop, dis, "healthy" in dis.lower()


def predict_both(img_pil):
    """Run both Standard CNN (256 features) and CNN+BDBO (107 features)."""
    img = img_pil.convert("RGB").resize((128, 128))
    arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, 0)

    # Standard CNN
    t0 = time.perf_counter()
    cnn_p = CNN.predict(arr, verbose=0)[0]
    t_cnn = (time.perf_counter() - t0) * 1000

    # CNN + BDBO
    t1 = time.perf_counter()
    feat = FEAT_EXT.predict(arr, verbose=0)

    if DBO_IDX.size == 0:
        raise RuntimeError("DBO_feature_indices.npy is empty; cannot run CNN+BDBO inference.")
    if np.any(DBO_IDX < 0) or np.any(DBO_IDX >= feat.shape[1]):
        raise RuntimeError(
            "DBO feature indices are outside the feature extractor output range. "
            f"Feature width: {feat.shape[1]}, index range: "
            f"{int(DBO_IDX.min())}..{int(DBO_IDX.max())}."
        )

    dbo_p = DBO.predict(feat[:, DBO_IDX], verbose=0)[0]
    t_dbo = (time.perf_counter() - t1) * 1000

    if len(CLASS_NAMES) != len(cnn_p) or len(CLASS_NAMES) != len(dbo_p):
        raise RuntimeError(
            "Model output dimensions do not match class_names.json: "
            f"{len(CLASS_NAMES)} classes, CNN output {len(cnn_p)}, "
            f"CNN+BDBO output {len(dbo_p)}."
        )

    return {
        "cnn_idx": int(np.argmax(cnn_p)),
        "cnn_conf": float(np.max(cnn_p)),
        "cnn_probs": cnn_p,
        "cnn_ms": t_cnn,
        "dbo_idx": int(np.argmax(dbo_p)),
        "dbo_conf": float(np.max(dbo_p)),
        "dbo_probs": dbo_p,
        "dbo_ms": t_dbo,
    }


def save_log(image_name, prediction, confidence):
    row = pd.DataFrame({
        "Timestamp": [pd.Timestamp.now()],
        "Image": [image_name],
        "Prediction": [prediction],
        "Confidence": [f"{confidence:.2%}"],
    })
    path = os.path.join(BASE_DIR, "prediction_logs.csv")
    if os.path.exists(path):
        row.to_csv(path, mode="a", header=False, index=False)
    else:
        row.to_csv(path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Session State & Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Home"

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
        <span style="font-size:2rem;">🌿</span>
        <div>
            <h3 style="margin:0;font-size:1.2rem;font-weight:800;color:#34d399;">LeafMedic AI</h3>
            <p style="margin:0;font-size:0.72rem;color:#94a3b8;">CNN + BDBO Plant Disease Detection</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🗂️ Navigation")

    if st.button("🏠  Home"):
        st.session_state.page = "Home"
        st.rerun()
    if st.button("🔬  Disease Recognition"):
        st.session_state.page = "Disease Recognition"
        st.rerun()
    if st.button("📖  Disease Lookup"):
        st.session_state.page = "Disease Lookup"
        st.rerun()
    if st.button("📊  Model Performance"):
        st.session_state.page = "Model Performance"
        st.rerun()
    if st.button("📋  Prediction Logs"):
        st.session_state.page = "Logs"
        st.rerun()
    if st.button("ℹ️  About"):
        st.session_state.page = "About"
        st.rerun()

    st.markdown("---")

    if OK:
        st.markdown("""
        <div style="background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);
                    border-radius:10px;padding:10px;margin-top:6px;">
            <div style="font-size:0.75rem;font-weight:700;color:#34d399;">● MODELS ONLINE</div>
            <div style="font-size:0.72rem;color:#cbd5e1;margin-top:4px;">
                CNN: 256 Features &nbsp;|&nbsp; CNN+BDBO: 107 Features<br>
                38 Disease Classes &nbsp;|&nbsp; 14 Crop Species
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ MODELS OFFLINE")
        with st.expander("Show loading error"):
            st.code(LOAD_ERR)
        if st.button("🔄 Retry Loading Models", key="sidebar_retry"):
            st.cache_resource.clear()
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE : HOME
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Home":
    st.markdown("""
    <div class="hero">
        <h1>🌿 LeafMedic: Plant Disease Detection System</h1>
        <p>
            Upload a leaf image and instantly diagnose plant diseases using our
            hybrid <b>CNN + Binary Dung Beetle Optimization (BDBO)</b> model.
            We compare a standard 256-feature CNN against our optimized
            107-feature CNN+BDBO pipeline — achieving <b>98.47 % accuracy</b>
            on 38 disease categories across 14 crop species.
        </p>
        <div class="pill-row">
            <span class="pill">🎯 98.47 % Accuracy</span>
            <span class="pill">⚡ 58.2 % Feature Reduction</span>
            <span class="pill">🌾 38 Classes</span>
            <span class="pill">🍎 14 Crops</span>
            <span class="pill">📈 +8.66 % Improvement over CNN</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### How It Works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:2.5rem;">📸</div>
            <h4 style="color:#064e3b;">1. Upload Image</h4>
            <p style="font-size:0.88rem;color:#64748b;">
                Go to <b>Disease Recognition</b> and upload or capture a photo of the affected leaf.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:2.5rem;">🧠</div>
            <h4 style="color:#064e3b;">2. AI Analysis</h4>
            <p style="font-size:0.88rem;color:#64748b;">
                Both the standard CNN and the proposed CNN+BDBO model run simultaneously for a side-by-side comparison.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:2.5rem;">💊</div>
            <h4 style="color:#064e3b;">3. Diagnosis & Treatment</h4>
            <p style="font-size:0.88rem;color:#64748b;">
                View the predicted disease, confidence scores, symptoms, and actionable management tips.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Why CNN + BDBO ?")
    st.markdown("""
    <div class="card card-accent">
        <p style="font-size:0.92rem;color:#334155;line-height:1.6;">
            Standard CNNs extract <b>256 deep features</b> but many capture background noise,
            lighting artifacts, or redundant patterns. Our <b>Binary Dung Beetle Optimization</b>
            algorithm — inspired by dung beetle foraging, rolling, and stealing behaviors —
            intelligently selects only the <b>107 most disease-discriminative features</b>,
            removing 149 redundant dimensions (58.2 % reduction). This yields:
        </p>
        <ul style="font-size:0.92rem;color:#334155;line-height:1.7;">
            <li>📈 <b>+8.66 %</b> accuracy improvement (89.81 % → 98.47 %)</li>
            <li>⚡ <b>Faster inference</b> with a smaller feature vector</li>
            <li>🛡️ <b>Better generalization</b> — less overfitting on unseen images</li>
            <li>💻 <b>Edge-friendly</b> — suitable for drones & IoT field sensors</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE : DISEASE RECOGNITION  (core prediction page)
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Disease Recognition":
    st.markdown("""
    <div class="hero">
        <h1>🔬 Disease Recognition</h1>
        <p>Upload or capture a leaf image. The system runs <b>two models</b>
        side-by-side and provides a complete diagnostic report.</p>
    </div>
    """, unsafe_allow_html=True)

    if not OK:
        st.error(f"⚠️ Models could not be loaded:\n\n{LOAD_ERR}")
        if st.button("🔄 Retry Loading Models", key="rec_retry"):
            st.cache_resource.clear()
            st.rerun()
        st.stop()

    # --- Input ---
    input_method = st.radio("Choose Input Method:", ["📤 Upload Image", "📷 Use Camera"], horizontal=True)
    image = None
    img_name = ""

    if input_method == "📤 Upload Image":
        uploaded = st.file_uploader("Select a leaf image (JPG / PNG):", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded)
            img_name = uploaded.name
    else:
        cam = st.camera_input("Capture a leaf image:")
        if cam:
            image = Image.open(cam)
            img_name = "Camera Capture"

    if image:
        col_img, col_info = st.columns([1, 1.5], gap="large")
        with col_img:
            st.image(image, caption=img_name, use_container_width=True)

    # --- Predict ---
    if st.button("🚀 Analyze Leaf", type="primary", use_container_width=True):
        if image is None:
            st.error("⚠️ Please provide an image first.")
            st.stop()

        with st.spinner("Running CNN & CNN+BDBO inference …"):
            res = predict_both(image)

        cnn_label = CLASS_NAMES[res["cnn_idx"]]
        dbo_label = CLASS_NAMES[res["dbo_idx"]]
        cnn_crop, cnn_dis, cnn_ok = _parse(cnn_label)
        dbo_crop, dbo_dis, dbo_ok = _parse(dbo_label)

        st.balloons()

        # ── Side-by-Side Result Cards ──
        st.markdown("---")
        st.markdown("### 📊 Dual-Model Comparison")

        col_cnn, col_dbo = st.columns(2, gap="large")

        with col_cnn:
            st.markdown(f"""
            <div class="card" style="border-top:4px solid #3b82f6;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                    <span style="background:#eff6ff;color:#1d4ed8;padding:4px 12px;border-radius:9999px;
                                 font-size:0.78rem;font-weight:700;">STANDARD CNN</span>
                    <span style="font-size:0.78rem;color:#64748b;">⚡ {res['cnn_ms']:.0f} ms &nbsp;|&nbsp; 256 Features</span>
                </div>
                <div style="font-size:0.8rem;color:#64748b;font-weight:600;">PREDICTED DISEASE</div>
                <div style="font-size:1.3rem;font-weight:800;color:#1e293b;">{cnn_crop}: {cnn_dis}</div>
                <div style="font-size:1.05rem;font-weight:700;color:#3b82f6;margin-top:8px;">
                    Confidence: {res['cnn_conf']*100:.2f} %
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Top-5 bar chart
            idx5 = np.argsort(res["cnn_probs"])[::-1][:5]
            fig1 = go.Figure(go.Bar(
                x=[res["cnn_probs"][i]*100 for i in idx5][::-1],
                y=[CLASS_NAMES[i].replace("___"," – ").replace("_"," ") for i in idx5][::-1],
                orientation="h",
                marker_color="#3b82f6",
                text=[f"{res['cnn_probs'][i]*100:.1f}%" for i in idx5][::-1],
                textposition="auto",
            ))
            fig1.update_layout(title="Top-5 Probabilities (CNN)", height=230,
                               margin=dict(l=10,r=20,t=35,b=20),
                               xaxis=dict(range=[0,105]),
                               paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True)

        with col_dbo:
            st.markdown(f"""
            <div class="card" style="border-top:4px solid #10b981;box-shadow:0 8px 24px rgba(16,185,129,0.12);">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                    <span style="background:#ecfdf5;color:#047857;padding:4px 12px;border-radius:9999px;
                                 font-size:0.78rem;font-weight:700;">★ PROPOSED CNN + BDBO</span>
                    <span style="font-size:0.78rem;color:#047857;">⚡ {res['dbo_ms']:.0f} ms &nbsp;|&nbsp; 107 Features (−58 %)</span>
                </div>
                <div style="font-size:0.8rem;color:#047857;font-weight:600;">PREDICTED DISEASE</div>
                <div style="font-size:1.3rem;font-weight:800;color:#064e3b;">{dbo_crop}: {dbo_dis}</div>
                <div style="font-size:1.05rem;font-weight:700;color:#059669;margin-top:8px;">
                    Confidence: {res['dbo_conf']*100:.2f} %
                </div>
            </div>
            """, unsafe_allow_html=True)

            idx5d = np.argsort(res["dbo_probs"])[::-1][:5]
            fig2 = go.Figure(go.Bar(
                x=[res["dbo_probs"][i]*100 for i in idx5d][::-1],
                y=[CLASS_NAMES[i].replace("___"," – ").replace("_"," ") for i in idx5d][::-1],
                orientation="h",
                marker_color="#10b981",
                text=[f"{res['dbo_probs'][i]*100:.1f}%" for i in idx5d][::-1],
                textposition="auto",
            ))
            fig2.update_layout(title="Top-5 Probabilities (CNN+BDBO)", height=230,
                               margin=dict(l=10,r=20,t=35,b=20),
                               xaxis=dict(range=[0,105]),
                               paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

        # ── Detailed Disease Report (uses CNN+BDBO result) ──
        st.markdown("---")
        info = DISEASE_INFO.get(dbo_label, {
            "crop": dbo_crop, "disease": dbo_dis,
            "pathogen": "Identified via deep learning",
            "symptoms": "Leaf discoloration / morphological anomaly detected.",
            "tips": ["Consult local agricultural extension for management advice."],
        })

        status_cls = "result-healthy" if dbo_ok else "result-disease"
        status_txt = "✅ HEALTHY — No Disease Detected" if dbo_ok else "⚠️ DISEASE DETECTED — Action Required"
        status_ico = "🟢" if dbo_ok else "🔴"

        st.markdown(f"""
        <div class="{status_cls}">
            <div style="font-size:1.2rem;font-weight:800;margin-bottom:8px;">{status_ico} {status_txt}</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
                <div><span style="font-weight:700;color:#64748b;">Crop:</span> {info['crop']}</div>
                <div><span style="font-weight:700;color:#64748b;">Disease:</span> {info['disease']}</div>
                <div><span style="font-weight:700;color:#64748b;">Pathogen:</span> {info['pathogen']}</div>
                <div><span style="font-weight:700;color:#64748b;">Confidence:</span> {res['dbo_conf']*100:.2f} %</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Symptoms
        st.markdown("#### 🔍 Symptoms")
        st.info(info["symptoms"])

        # Management Tips
        st.markdown("#### 💊 Management & Treatment Tips")
        for i, tip in enumerate(info["tips"], 1):
            st.markdown(f"""<div class="tip-item">✅ <b>Tip {i}:</b> {tip}</div>""", unsafe_allow_html=True)

        # Save log
        save_log(img_name, dbo_label, res["dbo_conf"])

        # Download Results
        st.markdown("---")
        report_text = (
            f"LeafMedic AI — Diagnostic Report\n"
            f"{'='*40}\n"
            f"Date: {pd.Timestamp.now()}\n"
            f"Image: {img_name}\n\n"
            f"STANDARD CNN:\n"
            f"  Prediction: {cnn_crop} — {cnn_dis}\n"
            f"  Confidence: {res['cnn_conf']*100:.2f}%\n"
            f"  Features Used: 256\n\n"
            f"PROPOSED CNN+BDBO:\n"
            f"  Prediction: {dbo_crop} — {dbo_dis}\n"
            f"  Confidence: {res['dbo_conf']*100:.2f}%\n"
            f"  Features Used: 107 (58.2% reduction)\n\n"
            f"PATHOGEN: {info['pathogen']}\n"
            f"SYMPTOMS: {info['symptoms']}\n\n"
            f"MANAGEMENT TIPS:\n"
        )
        for i, tip in enumerate(info["tips"], 1):
            report_text += f"  {i}. {tip}\n"

        st.download_button(
            "📥 Download Diagnostic Report",
            report_text.encode("utf-8"),
            file_name="leafmedic_report.txt",
            mime="text/plain",
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE : DISEASE LOOKUP
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Disease Lookup":
    st.markdown("""
    <div class="hero">
        <h1>📖 Disease Encyclopedia</h1>
        <p>Browse all 38 supported disease categories. Select a crop and condition to view
        symptoms, pathogen info, and management protocols.</p>
    </div>
    """, unsafe_allow_html=True)

    if not OK:
        st.error(f"⚠️ Models could not be loaded:\n\n{LOAD_ERR}")
        if st.button("🔄 Retry Loading Models", key="lookup_retry"):
            st.cache_resource.clear()
            st.rerun()
        st.stop()

    crops = sorted(set(_parse(c)[0] for c in CLASS_NAMES))
    sel_crop = st.selectbox("🌾 Filter by Crop", ["All Crops"] + crops)

    filtered = []
    for c in CLASS_NAMES:
        crop, dis, healthy = _parse(c)
        if sel_crop != "All Crops" and crop != sel_crop:
            continue
        filtered.append(c)

    sel_disease = st.selectbox("🦠 Select Disease", filtered,
                               format_func=lambda x: x.replace("___", " — ").replace("_", " "))

    if sel_disease:
        info = DISEASE_INFO.get(sel_disease, {
            "crop": _parse(sel_disease)[0],
            "disease": _parse(sel_disease)[1],
            "pathogen": "–",
            "symptoms": "–",
            "tips": ["Consult local extension."],
        })
        crop, dis, healthy = _parse(sel_disease)

        badge_bg = "#ecfdf5" if healthy else "#fef2f2"
        badge_col = "#047857" if healthy else "#b91c1c"
        badge_txt = "HEALTHY" if healthy else "DISEASED"

        st.markdown(f"""
        <div class="card" style="margin-top:14px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
                <h3 style="margin:0;color:#0f172a;">{crop}: {dis}</h3>
                <span style="background:{badge_bg};color:{badge_col};padding:4px 12px;
                             border-radius:9999px;font-size:0.78rem;font-weight:700;">{badge_txt}</span>
            </div>
            <div style="margin-bottom:12px;">
                <span style="font-weight:700;color:#475569;">Pathogen:</span>
                <span style="color:#334155;">{info['pathogen']}</span>
            </div>
            <div style="margin-bottom:16px;">
                <span style="font-weight:700;color:#475569;">Symptoms:</span>
                <span style="color:#334155;">{info['symptoms']}</span>
            </div>
            <h4 style="color:#064e3b;margin-bottom:8px;">💊 Management Tips</h4>
        </div>
        """, unsafe_allow_html=True)

        for i, tip in enumerate(info["tips"], 1):
            st.markdown(f"""<div class="tip-item">✅ <b>Tip {i}:</b> {tip}</div>""",
                        unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE : MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Model Performance":
    st.markdown("""
    <div class="hero">
        <h1>📊 Model Performance & Benchmarks</h1>
        <p>Comprehensive comparison of Standard CNN vs Proposed CNN+BDBO on the
        New Plant Diseases Dataset (87,867 images, 38 classes, 14 crop families).</p>
    </div>
    """, unsafe_allow_html=True)

    if not OK:
        st.error(f"⚠️ Models could not be loaded:\n\n{LOAD_ERR}")
        if st.button("🔄 Retry Loading Models", key="perf_retry"):
            st.cache_resource.clear()
            st.rerun()
        st.stop()

    # Dataset Stats
    st.markdown("#### 📦 Dataset")
    st.markdown("""
    <div class="m-grid">
        <div class="m-box"><div class="label">Total Images</div><div class="value">87,867</div><div class="sub">RGB Leaf Photographs</div></div>
        <div class="m-box"><div class="label">Training</div><div class="value">56,236</div><div class="sub">64 % Split</div></div>
        <div class="m-box"><div class="label">Validation</div><div class="value">14,059</div><div class="sub">16 % Split</div></div>
        <div class="m-box"><div class="label">Testing</div><div class="value">17,572</div><div class="sub">20 % Unseen</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Comparison Table
    st.markdown("#### ⚖️ Head-to-Head Comparison")
    df_cmp = pd.DataFrame({
        "Metric": ["Feature Dimension", "Feature Reduction", "Accuracy",
                    "Precision", "Recall", "F1-Score"],
        "Standard CNN": ["256", "0 %", "89.81 %", "~89 %", "~90 %", "~89 %"],
        "CNN + BDBO (Proposed)": ["107", "58.20 %", "98.47 %", "98.48 %", "98.47 %", "98.47 %"],
        "Improvement": ["-149 features", "+58.20 % lighter", "+8.66 %",
                        "+9.48 %", "+8.47 %", "+9.47 %"],
    })
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)

    # Bar Chart
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Standard CNN", x=metrics, y=[89.81,89,90,89], marker_color="#94a3b8"))
    fig.add_trace(go.Bar(name="CNN + BDBO", x=metrics, y=[98.47,98.48,98.47,98.47], marker_color="#10b981"))
    fig.update_layout(barmode="group", yaxis=dict(range=[0,110]), height=320,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # Feature Mask
    st.markdown("#### 🪲 BDBO Feature Selection Mask (256 Dimensions)")
    st.markdown("Green = Retained (107), Gray = Pruned (149)")
    mask_values = np.asarray(DBO_MASK).reshape(-1)
    mask_size = len(mask_values)
    colors = ["#10b981" if bool(m) else "#cbd5e1" for m in mask_values]
    fig_m = go.Figure(go.Bar(x=list(range(mask_size)), y=[1] * mask_size, marker_color=colors,
                             hovertext=[f"#{i}: {'KEEP' if bool(m) else 'PRUNE'}" for i, m in enumerate(mask_values)],
                             hoverinfo="text"))
    fig_m.update_layout(height=180, margin=dict(l=20,r=20,t=10,b=30),
                        yaxis=dict(visible=False), xaxis_title="Feature Index",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_m, use_container_width=True)

    # Training Curves
    if HIST and "accuracy" in HIST and "loss" in HIST:
        st.markdown("#### 📈 Training History")
        epochs = list(range(1, len(HIST["accuracy"])+1))
        c1, c2 = st.columns(2)
        with c1:
            fa = go.Figure()
            fa.add_trace(go.Scatter(x=epochs, y=[a*100 for a in HIST["accuracy"]],
                                    mode="lines+markers", name="Train Acc", line=dict(color="#10b981",width=3)))
            if "val_accuracy" in HIST:
                fa.add_trace(go.Scatter(x=epochs, y=[a*100 for a in HIST["val_accuracy"]],
                                        mode="lines+markers", name="Val Acc", line=dict(color="#3b82f6",width=3,dash="dash")))
            fa.update_layout(title="Accuracy", xaxis_title="Epoch", yaxis_title="%", height=280,
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fa, use_container_width=True)
        with c2:
            fl = go.Figure()
            fl.add_trace(go.Scatter(x=epochs, y=HIST["loss"],
                                    mode="lines+markers", name="Train Loss", line=dict(color="#ef4444",width=3)))
            if "val_loss" in HIST:
                fl.add_trace(go.Scatter(x=epochs, y=HIST["val_loss"],
                                        mode="lines+markers", name="Val Loss", line=dict(color="#f59e0b",width=3,dash="dash")))
            fl.update_layout(title="Loss", xaxis_title="Epoch", yaxis_title="Loss", height=280,
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fl, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE : LOGS
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Logs":
    st.markdown("""
    <div class="hero">
        <h1>📋 Prediction Logs</h1>
        <p>View and download all past prediction results.</p>
    </div>
    """, unsafe_allow_html=True)

    log_path = os.path.join(BASE_DIR, "prediction_logs.csv")
    if os.path.exists(log_path):
        logs = pd.read_csv(log_path)
        st.dataframe(logs, use_container_width=True, hide_index=True)
        st.download_button("📥 Download Logs CSV",
                           logs.to_csv(index=False).encode("utf-8"),
                           file_name="prediction_logs.csv")
    else:
        st.info("No predictions logged yet. Go to **Disease Recognition** to analyze a leaf image.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE : ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "About":
    st.markdown("""
    <div class="hero">
        <h1>ℹ️ About This Project</h1>
        <p>Technical details, dataset information, and research motivation.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🌍 Research Motivation

    Crop diseases cause **20–40 % annual yield losses** worldwide. Traditional expert-based
    inspection is slow, subjective, and impossible to scale. Deep learning CNNs can automate
    detection but standard models suffer from:

    - **Redundant features** — noise, background artifacts, lighting patterns
    - **Overfitting risks** — models memorize training-specific details
    - **High compute needs** — impractical for edge / IoT deployment

    ### 🚀 Our Hybrid CNN + BDBO Solution

    | Component | Role |
    |---|---|
    | **CNN Backbone** | 4 Conv blocks with Batch Norm, ReLU, MaxPool, GAP → **256-D feature vector** |
    | **BDBO Optimizer** | Bio-inspired swarm algorithm selects **107 optimal features** (58.2 % reduction) |
    | **Dense Classifier** | 128-unit Dense + Dropout → 38-class Softmax output |

    ### 📦 Dataset: New Plant Diseases Dataset

    | Split | Images |
    |---|---:|
    | Training | 56,236 |
    | Validation | 14,059 |
    | Testing | 17,572 |
    | **Total** | **87,867** |
    | Classes | 38 |
    | Image Size | 128 × 128 × 3 RGB |

    ### 📊 Key Results

    | Metric | Standard CNN | CNN + BDBO | Δ |
    |---|---:|---:|---:|
    | Accuracy | 89.81 % | **98.47 %** | +8.66 % |
    | Precision | ~89 % | **98.48 %** | +9.48 % |
    | Recall | ~90 % | **98.47 %** | +8.47 % |
    | F1-Score | ~89 % | **98.47 %** | +9.47 % |
    | Features | 256 | 107 | −58.2 % |

    ### 🪲 How BDBO Works

    1. **Rolling Phase** — Broad exploration across the binary feature space
    2. **Dancing Phase** — Orientation adjustments when stuck at local optima
    3. **Breeding Phase** — Adaptive boundary construction around promising subsets
    4. **Foraging Phase** — Fine-grained local search toggling individual feature bits
    5. **Stealing Phase** — Rapid convergence toward the global optimum

    ### 🛠️ Tech Stack

    - **Models:** TensorFlow / Keras
    - **Optimization:** Custom BDBO (Binary Dung Beetle Optimization)
    - **Frontend:** Streamlit + Plotly
    - **Dataset:** Kaggle — New Plant Diseases Dataset
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:0.8rem;color:#94a3b8;padding:8px 0;">
    🌿 <b>LeafMedic AI</b> — CNN + Binary Dung Beetle Optimization •
    87,867 Images • 38 Pathologies • 14 Crops •
    Built with Streamlit, TensorFlow & Plotly
</div>
""", unsafe_allow_html=True)
