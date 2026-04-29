# ============================================================
# BagginessControl — Dashboard Industrial de Calidad de Papel
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import io
import os
import tempfile

# ─── Constants ───────────────────────────────────────────────────────────────
MODEL_PATH       = "model_bagginess.pkl"       # Excel-trained model
JSON_MODEL_PATH  = "uploaded_model.json"        # JSON-uploaded model
TRAIN_INFO_PATH  = "train_info.json"            # Metrics from Excel training
JSON_INFO_PATH   = "json_info.json"             # Info about JSON model
CONFIG_FILE      = "bagginess_config.json"
FEATURES    = ["Variabilidad", "Simetría", "Curvatura", "STDEV",
               "M5", "M1", "M3", "M4", "Simetría L3"]
DUREZA_COLS = [f"Dureza Rollo ({i})" for i in range(1, 21)]

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BagginessControl | Calidad de Papel",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@300;400;500;600;700&family=Fira+Code:wght@400;500;600&display=swap');
@import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Fira Sans', 'Segoe UI', sans-serif; }
.stApp { background-color: #F4F7FB; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Sidebar width ── */
section[data-testid="stSidebar"] { width: 210px !important; min-width: 210px !important; }
section[data-testid="stSidebar"] > div:first-child { width: 210px !important; padding-left: 0.9rem; padding-right: 0.9rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f35 0%, #0f2847 40%, #132d52 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #e8f2ff !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown small { color: #d4e6f8 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.18); margin: 0.8rem 0; }
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label,
[data-testid="stSidebar"] .stRadio p { color: #e8f2ff !important; font-weight: 500 !important; }
[data-testid="stSidebar"] .stSuccess {
    background: rgba(39,174,96,0.22) !important;
    border: 1px solid rgba(39,174,96,0.45) !important;
    color: #7dffb3 !important;
}
[data-testid="stSidebar"] .stSuccess * { color: #7dffb3 !important; }
[data-testid="stSidebar"] .stWarning {
    background: rgba(243,156,18,0.18) !important;
    border: 1px solid rgba(243,156,18,0.38) !important;
    color: #ffd97a !important;
}
[data-testid="stSidebar"] .stWarning * { color: #ffd97a !important; }

/* ── Section header ── */
.section-hdr {
    background: linear-gradient(90deg, #0d1f35 0%, #1557a0 100%);
    color: white;
    padding: 14px 22px;
    border-radius: 10px;
    margin-bottom: 22px;
    box-shadow: 0 3px 12px rgba(13,31,53,0.25);
}
.section-hdr h2 {
    color: white !important;
    margin: 0;
    font-size: 1.25em;
    font-weight: 600;
    letter-spacing: 0.3px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-hdr h2 i { font-size: 1.05em; opacity: 0.92; }

/* ── Headings ── */
h1, h2, h3 { color: #0d1f35 !important; font-family: 'Fira Sans', sans-serif !important; }
h3 { font-size: 1.1em !important; font-weight: 600 !important; margin-bottom: 12px !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: white;
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border-top: 3px solid #1557a0;
}
[data-testid="stMetricLabel"] { font-size: 0.78em !important; color: #6b7a8d !important; font-weight: 500; }
[data-testid="stMetricValue"] { font-size: 1.6em !important; font-weight: 700 !important; color: #0d1f35 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1557a0, #0d3a6e) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 11px 24px !important;
    font-weight: 600 !important;
    font-family: 'Fira Sans', sans-serif !important;
    font-size: 0.95em !important;
    letter-spacing: 0.3px;
    transition: all 0.2s ease;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1a6eb5, #1557a0) !important;
    box-shadow: 0 4px 14px rgba(21,87,160,0.4) !important;
    transform: translateY(-1px);
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #1a7a4a, #145c36) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 11px 24px !important;
    font-weight: 600 !important;
    font-size: 0.95em !important;
    cursor: pointer !important;
}
.stDownloadButton > button:hover {
    box-shadow: 0 4px 14px rgba(26,122,74,0.4) !important;
    transform: translateY(-1px);
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: white;
    border-radius: 10px;
    padding: 14px;
    border: 2px dashed #1557a0;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 18px !important;
    background: #e4ecf7 !important;
    color: #1557a0 !important;
    font-weight: 500 !important;
    font-size: 0.92em !important;
    border: none !important;
    cursor: pointer;
}
.stTabs [aria-selected="true"] {
    background: #1557a0 !important;
    color: white !important;
    font-weight: 600 !important;
}

/* ── Cards ── */
.info-card {
    background: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    margin-bottom: 16px;
}
.spec-card {
    background: #f0f5ff;
    border-radius: 10px;
    padding: 18px 22px;
    border-left: 4px solid #1557a0;
    margin-bottom: 18px;
    font-size: 0.9em;
    color: #2a3f55;
}
.spec-card h4 { color: #0d1f35 !important; font-size: 0.95em !important; margin: 0 0 10px 0 !important; font-weight: 600; }
.spec-card code { background: #dce8f8; padding: 2px 6px; border-radius: 4px; font-size: 0.88em; }

/* ── Model selector card ── */
.model-sel-card {
    background: white;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border-left: 4px solid #1557a0;
    margin-bottom: 20px;
}

/* ── Status badges ── */
.badge-rechazar {
    background: linear-gradient(135deg, #a8071a, #cf1322);
    color: white; border-radius: 14px; padding: 30px 24px;
    text-align: center; box-shadow: 0 8px 24px rgba(207,19,34,0.35);
    margin-bottom: 18px; border: 1px solid rgba(255,255,255,0.15);
}
.badge-alerta {
    background: linear-gradient(135deg, #d46b08, #fa8c16);
    color: white; border-radius: 14px; padding: 30px 24px;
    text-align: center; box-shadow: 0 8px 24px rgba(250,140,22,0.35);
    margin-bottom: 18px; border: 1px solid rgba(255,255,255,0.15);
}
.badge-ok {
    background: linear-gradient(135deg, #096dd9, #1890ff);
    color: white; border-radius: 14px; padding: 30px 24px;
    text-align: center; box-shadow: 0 8px 24px rgba(24,144,255,0.35);
    margin-bottom: 18px; border: 1px solid rgba(255,255,255,0.15);
}
.badge-icon { font-size: 2.6em; margin-bottom: 8px; line-height: 1; }
.badge-icon i { font-size: inherit; }
.badge-text { font-size: 2.4em; font-weight: 900; letter-spacing: 3px; }
.badge-sub  { font-size: 0.95em; opacity: 0.9; margin-top: 8px; line-height: 1.4; }

/* ── Text area ── */
.stTextArea textarea {
    font-family: 'Fira Code', monospace !important;
    font-size: 0.92em !important;
    border-radius: 8px !important;
    border: 1.5px solid #d0dbe8 !important;
    background: #fafcff !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }

/* ── Progress ── */
.stProgress > div > div { background: linear-gradient(90deg, #1557a0, #1890ff); border-radius: 4px; }

/* ── Placeholder card ── */
.placeholder-card {
    background: white; border-radius: 14px; padding: 48px 32px;
    text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 2px dashed #ccd8e8; margin-top: 8px;
}
.placeholder-icon { font-size: 4em; margin-bottom: 16px; color: #8099b8; line-height: 1; }
.placeholder-icon i { font-size: inherit; }
.placeholder-title { font-size: 1.15em; color: #3a4f6a; font-weight: 600; margin-bottom: 10px; }
.placeholder-desc { font-size: 0.9em; color: #8095aa; line-height: 1.6; }

/* ── Login card ── */
.login-card {
    background: white; border-radius: 14px; padding: 40px 36px;
    max-width: 420px; margin: 60px auto 0;
    box-shadow: 0 4px 24px rgba(0,0,0,0.09);
    border-top: 4px solid #1557a0;
    text-align: center;
}
.login-card h3 { font-size: 1.2em !important; color: #0d1f35 !important; margin-bottom: 6px !important; }
.login-card p  { font-size: 0.88em; color: #6b7a8d; margin-bottom: 24px; }

/* ── Sidebar logo ── */
.sidebar-logo {
    text-align: center; padding: 12px 0 4px;
    border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 14px;
}
.sidebar-logo .logo-icon { font-size: 1.6em; color: #a8c8f0; margin-bottom: 4px; }
.sidebar-logo .logo-text { font-size: 1.15em; font-weight: 700; color: #e0ecff !important; letter-spacing: 0.5px; }
.sidebar-logo .logo-sub  { font-size: 0.72em; color: #8aaac8 !important; margin-top: 2px; }

/* ── Alert/Success ── */
.stSuccess { border-radius: 8px !important; }
.stWarning { border-radius: 8px !important; }
.stError   { border-radius: 8px !important; }
.stInfo    { border-radius: 8px !important; }

/* ── Responsive ── */
@media screen and (max-width: 900px) {
    .block-container { padding-left: 0.6rem !important; padding-right: 0.6rem !important; }
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
    [data-testid="stColumn"] { min-width: min(100%, 280px) !important; flex: 1 1 280px !important; }
    .badge-text { font-size: 1.8em !important; }
    .badge-icon { font-size: 2em !important; }
    .section-hdr h2 { font-size: 1.05em !important; }
}
@media screen and (max-width: 600px) {
    [data-testid="stMetricValue"] { font-size: 1.2em !important; }
    .login-card { padding: 28px 18px; margin-top: 20px; }
    .placeholder-card { padding: 28px 16px; }
}
</style>
""", unsafe_allow_html=True)

# ─── Config helpers ───────────────────────────────────────────────────────────

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# ─── Disk persistence ─────────────────────────────────────────────────────────

def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def initialize_from_disk():
    """Load models and metadata from disk into session state (runs once per session)."""
    if st.session_state.get("_disk_initialized"):
        return

    # Excel-trained model
    if os.path.exists(MODEL_PATH):
        try:
            m = joblib.load(MODEL_PATH)
            st.session_state.setdefault("excel_model", m)
        except Exception:
            pass

    # Excel training metrics
    if os.path.exists(TRAIN_INFO_PATH):
        try:
            st.session_state.setdefault("train_info", _load_json(TRAIN_INFO_PATH))
        except Exception:
            pass

    # JSON-uploaded model
    if os.path.exists(JSON_MODEL_PATH):
        try:
            m = XGBClassifier()
            m.load_model(JSON_MODEL_PATH)
            st.session_state.setdefault("json_model", m)
        except Exception:
            pass

    # JSON model info
    if os.path.exists(JSON_INFO_PATH):
        try:
            st.session_state.setdefault("json_info", _load_json(JSON_INFO_PATH))
        except Exception:
            pass

    # Default model source
    if "model_source" not in st.session_state:
        if st.session_state.get("json_model") is not None:
            st.session_state["model_source"] = "JSON cargado"
        elif st.session_state.get("excel_model") is not None:
            st.session_state["model_source"] = "entrenado"

    st.session_state["_disk_initialized"] = True


initialize_from_disk()


# ─── Helper functions ─────────────────────────────────────────────────────────

def calcular_variables(source):
    if isinstance(source, (list, np.ndarray)):
        arr = np.array(source, dtype=float)
    else:
        arr = source[DUREZA_COLS].values.astype(float)

    M5 = np.mean(arr[0:4]);  M4 = np.mean(arr[4:8])
    M3 = np.mean(arr[8:12]); M1 = np.mean(arr[16:20])

    Variabilidad = float(np.max(arr) - np.min(arr))
    STDEV        = float(np.std(arr))
    Simetria     = float(M5 - M1)
    Extremo      = float(np.mean(np.concatenate([arr[1:4], arr[16:19]])))
    Curvatura    = float(Extremo - M3)
    Simetria_L3  = float(M5 - M4)

    return pd.Series({
        "Variabilidad": Variabilidad, "Simetría": Simetria,
        "Curvatura": Curvatura,       "STDEV": STDEV,
        "M5": float(M5), "M1": float(M1),
        "M3": float(M3), "M4": float(M4),
        "Simetría L3": Simetria_L3,
    })


def clasificar(prob, stdev):
    if prob >= 0.80: return "RECHAZAR"
    if prob >= 0.60 and stdev > 3.3: return "RECHAZAR"
    if prob >= 0.50 or stdev > 3.0:  return "ALERTA"
    return "OK"


def load_model():
    src = st.session_state.get("model_source", "")
    if src == "JSON cargado":
        m = st.session_state.get("json_model")
        if m is None and os.path.exists(JSON_MODEL_PATH):
            try:
                m = XGBClassifier()
                m.load_model(JSON_MODEL_PATH)
                st.session_state["json_model"] = m
            except Exception:
                pass
        return m
    else:
        m = st.session_state.get("excel_model")
        if m is None and os.path.exists(MODEL_PATH):
            try:
                m = joblib.load(MODEL_PATH)
                st.session_state["excel_model"] = m
            except Exception:
                pass
        return m


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    return buf


def _render_confusion_report(cm, report, n_train=None, n_test=None,
                              n_train_0=None, n_train_1=None,
                              n_test_0=None,  n_test_1=None,
                              show_split=True):
    col_cm, col_rpt = st.columns(2)

    with col_cm:
        st.markdown("**Matriz de Confusión — 100 % de los datos**")
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        fig_cm.patch.set_facecolor("white")
        sns.heatmap(
            np.array(cm), annot=True, fmt="d",
            cmap=sns.color_palette("Blues", as_cmap=True),
            xticklabels=["Sin Reclamo", "Reclamado"],
            yticklabels=["Sin Reclamo", "Reclamado"],
            ax=ax_cm, linewidths=1.5, linecolor="white",
            annot_kws={"size": 15, "weight": "bold"}, cbar=False,
        )
        ax_cm.set_ylabel("Real",     fontsize=10, fontweight="bold")
        ax_cm.set_xlabel("Predicho", fontsize=10, fontweight="bold")
        ax_cm.set_title("Matriz de Confusión", fontsize=11, fontweight="bold", pad=8)
        ax_cm.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)

    with col_rpt:
        if show_split and n_train is not None:
            st.markdown("**Distribución de datos**")
            d1, d2 = st.columns(2)
            d1.metric("Train (75%)",
                      f"{n_train} reeles",
                      f"0→{n_train_0}  |  1→{n_train_1}")
            d2.metric("Test (25%) — no entrenado",
                      f"{n_test} reeles",
                      f"0→{n_test_0}  |  1→{n_test_1}")

        st.markdown("**Reporte de Clasificación** *(100 % datos)*")
        rows_r = []
        for cls_k, cls_lbl in [
            ("0", "Sin Reclamo (0)"), ("1", "Reclamado (1)"),
            ("macro avg", "Promedio Macro"), ("weighted avg", "Prom. Ponderado"),
        ]:
            r = report.get(cls_k, {})
            rows_r.append({
                "Clase"    : cls_lbl,
                "Precision": round(r.get("precision", 0), 3),
                "Recall"   : round(r.get("recall",    0), 3),
                "F1-score" : round(r.get("f1-score",  0), 3),
                "Soporte"  : int(r.get("support",     0)),
            })
        st.dataframe(pd.DataFrame(rows_r), use_container_width=True, hide_index=True)

        acc    = report.get("accuracy", 0)
        rec_1  = report.get("1", {}).get("recall",    0)
        prec_1 = report.get("1", {}).get("precision", 0)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Accuracy",              f"{acc:.1%}")
        mc2.metric("Recall (Reclamado)",    f"{rec_1:.1%}")
        mc3.metric("Precision (Reclamado)", f"{prec_1:.1%}")

        if rec_1 >= 0.80:
            st.success(f"Buena detección de reclamos — Recall = {rec_1:.1%}")
        elif rec_1 >= 0.60:
            st.warning(f"Recall moderado ({rec_1:.1%}) — considere ajustar umbral")
        else:
            st.error(f"Recall bajo en clase Reclamado ({rec_1:.1%})")
        if show_split and n_train_0 is not None:
            st.markdown(f"**Desbalanceo:** `{n_train_0}` sin reclamo vs `{n_train_1}` reclamados en entrenamiento")


def _render_risk_scatter(df_scores):
    fig_sc, axes_sc = plt.subplots(1, 3, figsize=(17, 5.5))
    fig_sc.patch.set_facecolor("white")
    fig_sc.suptitle(
        "Probabilidad de Reclamo vs Variables Clave del Proceso",
        fontsize=13, fontweight="bold", y=1.01,
    )
    for ax_s, (var, xlabel, title) in zip(axes_sc, [
        ("Variabilidad", "Variabilidad (max − min)",  "Riesgo vs Variabilidad"),
        ("Simetría",     "Simetría (M5 − M1)",        "Riesgo vs Simetría"),
        ("Curvatura",    "Curvatura (borde − centro)", "Riesgo vs Curvatura"),
    ]):
        for dec, color in CMAP_DEC.items():
            mask = df_scores["Decision"] == dec
            ax_s.scatter(
                df_scores.loc[mask, var], df_scores.loc[mask, "Prob_modelo"],
                c=color, label=dec, alpha=0.72,
                edgecolors="white", linewidths=0.6, s=68, zorder=3,
            )
        ax_s.axhline(y=0.50, color="#888", linestyle="--", alpha=0.55, lw=1.2)
        ax_s.axhline(y=0.80, color="#cf1322", linestyle="--", alpha=0.55, lw=1.2)
        x_max = df_scores[var].max()
        ax_s.text(x_max * 1.01, 0.51, "50 %", fontsize=7.5, color="#888",    va="bottom")
        ax_s.text(x_max * 1.01, 0.81, "80 %", fontsize=7.5, color="#cf1322", va="bottom")
        ax_s.set_xlabel(xlabel, fontsize=10)
        ax_s.set_ylabel("Prob. de Reclamo", fontsize=10)
        ax_s.set_title(title, fontsize=10.5, fontweight="bold")
        ax_s.set_ylim(-0.05, 1.08)
        ax_s.legend(fontsize=8, framealpha=0.75, loc="upper left")
    plt.tight_layout()
    st.pyplot(fig_sc, use_container_width=True)


def _render_feature_importance(importances_dict):
    importances = pd.Series(importances_dict).sort_values()
    fig_fi, ax_fi = plt.subplots(figsize=(10, 4))
    fig_fi.patch.set_facecolor("white")
    bar_colors = [
        "#cf1322" if v == importances.max()
        else "#1557a0" if v >= importances.quantile(0.66)
        else "#5b8cc8"
        for v in importances.values
    ]
    bars = ax_fi.barh(importances.index, importances.values,
                      color=bar_colors, height=0.62, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, importances.values):
        ax_fi.text(
            val + importances.max() * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9.5, color="#2a3f55",
        )
    ax_fi.set_xlabel("Importancia (F-score Gain)", fontsize=11)
    ax_fi.set_title("Variables más importantes para predecir Bagginess",
                    fontsize=12, fontweight="bold", pad=12)
    ax_fi.set_xlim(0, importances.max() * 1.20)
    ax_fi.grid(axis="x", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_fi, use_container_width=True)
    st.info(f"**Variable más importante: {importances.idxmax()}**")


# ─── Matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.facecolor": "#FAFCFF",
    "figure.facecolor": "white",  "axes.edgecolor": "#dce5ef",
    "axes.grid": True,            "grid.color": "#e8eff7",
    "grid.linestyle": "--",       "grid.linewidth": 0.7,
    "axes.spines.top": False,     "axes.spines.right": False,
    "xtick.color": "#445566",     "ytick.color": "#445566",
    "axes.labelcolor": "#2a3f55", "text.color": "#2a3f55",
})
CMAP_DEC = {"RECHAZAR": "#cf1322", "ALERTA": "#fa8c16", "OK": "#27ae60"}

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon"><i class="bi bi-activity"></i></div>
        <div class="logo-text">BagginessControl</div>
        <div class="logo-sub">Control de Calidad · Test Liner</div>
    </div>
    """, unsafe_allow_html=True)

    modo = st.radio(
        "Navegación",
        ["Administrador", "Predicción Reel", "Análisis Histórico"],
        label_visibility="collapsed",
    )

    st.markdown("---")




# =============================================================================
# PARTE 1 — ADMINISTRADOR / ENTRENAMIENTO
# =============================================================================
if "Administrador" in modo:

    # ── Password gate ─────────────────────────────────────────────────────────
    cfg          = load_config()
    has_password = bool(cfg.get("admin_password"))

    if has_password and not st.session_state.get("admin_auth"):
        st.markdown(
            '<div class="section-hdr"><h2>'
            '<i class="bi bi-sliders2-vertical"></i>'
            ' Administrador — Acceso restringido'
            '</h2></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown("### Acceso Administrador")
        st.markdown('<p>Ingrese la contraseña para acceder a la configuración del modelo.</p>',
                    unsafe_allow_html=True)
        pwd_input = st.text_input("Contraseña", type="password", key="pwd_input",
                                  label_visibility="collapsed",
                                  placeholder="Contraseña de administrador")
        btn_login = st.button("Acceder", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if btn_login:
            if pwd_input == cfg["admin_password"]:
                st.session_state["admin_auth"] = True
                st.rerun()
            else:
                st.error("Contraseña incorrecta")
        st.stop()

    # ── Admin header ──────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-hdr"><h2>'
        '<i class="bi bi-sliders2-vertical"></i>'
        ' Administrador — Modelo XGBoost'
        '</h2></div>',
        unsafe_allow_html=True,
    )

    # ── Model selector ────────────────────────────────────────────────────────
    has_excel_m = os.path.exists(MODEL_PATH) or st.session_state.get("excel_model") is not None
    has_json_m  = os.path.exists(JSON_MODEL_PATH) or st.session_state.get("json_model") is not None

    src_map = {}
    if has_excel_m: src_map["Entrenado (.xlsx)"] = "entrenado"
    if has_json_m:  src_map["Cargado (.json)"]   = "JSON cargado"


    if src_map:
        labels      = list(src_map.keys())
        current_src = st.session_state.get("model_source", list(src_map.values())[0])
        cur_lbl     = labels[0]
        for lbl, src in src_map.items():
            if src == current_src:
                cur_lbl = lbl
                break
        idx = labels.index(cur_lbl) if cur_lbl in labels else 0

        col_sel, col_status = st.columns([2, 3])
        with col_sel:
            st.markdown("**Modelo activo para predicción**")
            sel_lbl = st.selectbox(
                "Modelo activo", labels, index=idx,
                label_visibility="collapsed", key="admin_model_sel",
            )
        new_src = src_map[sel_lbl]
        st.session_state["model_source"] = new_src

        with col_status:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            if load_model() is not None:
                st.success(f"Activo — {new_src}")
            else:
                st.warning("Sin modelo disponible")
    else:
        st.info("No hay modelos disponibles todavía. Entrene uno con Excel o cargue un archivo .json.")


    st.markdown("---")
    tab_excel, tab_json = st.tabs(["Entrenar con Excel", "Cargar modelo (.json)"])

    # ── Tab 1: Train from Excel ───────────────────────────────────────────────
    with tab_excel:

        # Show persisted training results if available
        ti = st.session_state.get("train_info")
        if ti is not None and (st.session_state.get("excel_model") is not None or os.path.exists(MODEL_PATH)):
            col_ti_hdr, col_ti_del = st.columns([4, 1])
            with col_ti_hdr:
                st.success(f"Modelo entrenado — **{ti['n_total']}** reeles · desbalanceo {ti['ratio']:.1f}:1")
            with col_ti_del:
                if st.button("Eliminar modelo", key="del_excel", help="Elimina el modelo entrenado del disco"):
                    for p in [MODEL_PATH, TRAIN_INFO_PATH]:
                        if os.path.exists(p):
                            os.remove(p)
                    st.session_state.pop("excel_model", None)
                    st.session_state.pop("train_info", None)
                    if st.session_state.get("model_source") == "entrenado":
                        st.session_state.pop("model_source", None)
                    st.rerun()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Reeles",    ti["n_total"])
            c2.metric("Sin Reclamo (0)", ti["n_ok"])
            c3.metric("Reclamados (1)",  ti["n_rec"])
            c4.metric("Desbalanceo",     f"{ti['ratio']:.1f} : 1")

            tab_m, tab_i, tab_r = st.tabs(["Confusión y Métricas",
                                            "Importancia de Variables",
                                            "Gráficas de Riesgo"])
            with tab_m:
                _render_confusion_report(
                    ti["cm"], ti["report"],
                    n_train=ti["n_train"], n_test=ti["n_test"],
                    n_train_0=ti["n_train_0"], n_train_1=ti["n_train_1"],
                    n_test_0=ti["n_test_0"],  n_test_1=ti["n_test_1"],
                )
            with tab_i:
                _render_feature_importance(ti["importances"])
            with tab_r:
                _render_risk_scatter(pd.DataFrame(ti["df_scatter"]))

            st.markdown("---")
            st.markdown("#### Reentrenar con nuevo archivo")

        uploaded = st.file_uploader(
            "Cargar archivo Excel",
            type=["xlsx", "xls"],
            help="Columnas requeridas: 'Dureza Rollo (1)' al '(20)' y 'Reclamado' (0/1).",
        )

        if uploaded:
            try:
                sheet_names = pd.ExcelFile(uploaded).sheet_names
                uploaded.seek(0)
            except Exception as exc:
                st.error(f"No se pudo leer el archivo: {exc}"); st.stop()

            selected_sheet = st.selectbox(
                "Seleccionar hoja", options=sheet_names,
                index=sheet_names.index("Dureza (2)") if "Dureza (2)" in sheet_names else 0,
            )
            with st.spinner("Leyendo datos…"):
                try:
                    uploaded.seek(0)
                    df_raw = pd.read_excel(uploaded, sheet_name=selected_sheet)
                    df_raw.columns = df_raw.columns.str.strip()
                except Exception as exc:
                    st.error(f"Error al leer archivo: {exc}"); st.stop()

            missing_dur = [c for c in DUREZA_COLS if c not in df_raw.columns]
            if missing_dur:
                st.error(f"Faltan columnas de dureza: {missing_dur[:4]} …")
                st.info(f"Columnas detectadas: {list(df_raw.columns[:12])}")
                st.stop()
            if "Reclamado" not in df_raw.columns:
                st.error("Falta la columna **'Reclamado'** (0 = sin reclamo, 1 = reclamado).")
                st.stop()

            st.success(f"**{len(df_raw)}** reeles cargados correctamente")
            df_vars  = df_raw.apply(calcular_variables, axis=1)
            df_train = pd.concat([df_raw, df_vars], axis=1)
            df_train = df_train.loc[:, ~df_train.columns.duplicated(keep="first")]

            n_total = len(df_train)
            n_ok    = int((df_train["Reclamado"] == 0).sum())
            n_rec   = int((df_train["Reclamado"] == 1).sum())
            ratio   = n_ok / max(n_rec, 1)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Reeles",    n_total)
            c2.metric("Sin Reclamo (0)", n_ok)
            c3.metric("Reclamados (1)",  n_rec)
            c4.metric("Desbalanceo",     f"{ratio:.1f} : 1")

            with st.expander("Vista previa — Variables calculadas", expanded=False):
                preview = [c for c in FEATURES + ["Reclamado"] if c in df_train.columns]
                st.dataframe(df_train[preview].head(25).round(3), use_container_width=True)

            st.markdown("---")
            col_btn, _ = st.columns([1, 2])
            with col_btn:
                entrenar = st.button("Entrenar Modelo XGBoost", use_container_width=True)

            if entrenar:
                prog = st.progress(0, "Preparando datos…")
                df_model = df_train[FEATURES + ["Reclamado"]].dropna().copy()
                X = df_model[FEATURES]
                y = df_model["Reclamado"].astype(int)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42, stratify=y)
                n_train_0 = int((y_train == 0).sum())
                n_train_1 = int((y_train == 1).sum())
                prog.progress(25, "Entrenando XGBoost…")
                scale_pos_weight = n_train_0 / n_train_1 if n_train_1 > 0 else 1
                xgb_m = XGBClassifier(
                    n_estimators=400, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss", random_state=42, verbosity=0,
                )
                xgb_m.fit(X_train, y_train)
                prog.progress(65, "Calculando métricas…")
                df_scores = df_model.copy()
                df_scores["Prob_modelo"] = xgb_m.predict_proba(X)[:, 1]
                df_scores["Decision"]    = df_scores.apply(
                    lambda r: clasificar(r["Prob_modelo"], r["STDEV"]), axis=1)
                y_pred = (df_scores["Decision"] == "RECHAZAR").astype(int)
                report = classification_report(y, y_pred, output_dict=True, zero_division=0)
                cm     = confusion_matrix(y, y_pred)

                # Save model to disk
                joblib.dump(xgb_m, MODEL_PATH)

                train_info = {
                    "n_total": n_total, "n_ok": n_ok, "n_rec": n_rec, "ratio": ratio,
                    "n_train": len(y_train), "n_test": len(y_test),
                    "n_train_0": n_train_0, "n_train_1": n_train_1,
                    "n_test_0": int((y_test == 0).sum()), "n_test_1": int((y_test == 1).sum()),
                    "importances": dict(zip(FEATURES, xgb_m.feature_importances_.tolist())),
                    "cm": cm.tolist(), "report": report,
                    "df_scatter": df_scores[
                        ["Variabilidad", "Simetría", "Curvatura", "Prob_modelo", "Decision"]
                    ].to_dict("records"),
                }
                # Save metrics to disk so they survive restarts
                _save_json(TRAIN_INFO_PATH, train_info)

                st.session_state["excel_model"]  = xgb_m
                st.session_state["train_info"]   = train_info
                st.session_state["model_source"] = "entrenado"
                prog.progress(100, "Modelo entrenado y guardado"); prog.empty()
                st.rerun()

    # ── Tab 2: Load pre-trained JSON model ───────────────────────────────────
    with tab_json:

        # Show persisted JSON model info if available
        ji = st.session_state.get("json_info")
        if ji is not None and (st.session_state.get("json_model") is not None or os.path.exists(JSON_MODEL_PATH)):
            col_ji_hdr, col_ji_del = st.columns([4, 1])
            with col_ji_hdr:
                st.success(f"Modelo cargado — **{ji['n_trees']}** árboles")
            with col_ji_del:
                if st.button("Eliminar modelo", key="del_json", help="Elimina el modelo JSON del disco"):
                    for p in [JSON_MODEL_PATH, JSON_INFO_PATH]:
                        if os.path.exists(p):
                            os.remove(p)
                    st.session_state.pop("json_model", None)
                    st.session_state.pop("json_info", None)
                    st.session_state.pop("json_eval_info", None)
                    if st.session_state.get("model_source") == "JSON cargado":
                        st.session_state.pop("model_source", None)
                    st.rerun()

            with st.expander("Importancia de variables del modelo cargado", expanded=True):
                _render_feature_importance(ji["importances"])

            # Show persisted evaluation results
            ei = st.session_state.get("json_eval_info")
            if ei is not None:
                st.markdown("---")
                st.success(f"Evaluación: **{ei['n_rows']}** reeles")
                tab_ev_cm, tab_ev_sc = st.tabs(["Confusión y Métricas", "Gráficas de Riesgo"])
                with tab_ev_cm:
                    _render_confusion_report(ei["cm"], ei["report"], show_split=False)
                with tab_ev_sc:
                    _render_risk_scatter(pd.DataFrame(ei["df_scatter"]))

            st.markdown("---")
            st.markdown("#### Evaluar rendimiento con datos etiquetados")
            st.markdown(
                "Suba un Excel con las 20 columnas de dureza y la columna **Reclamado** (0/1) "
                "para simular la matriz de confusión y gráficas de riesgo."
            )
            excel_eval = st.file_uploader(
                "Cargar Excel de evaluación", type=["xlsx", "xls"],
                key="json_eval_excel",
                help="Requiere columnas 'Dureza Rollo (1)' al '(20)' y 'Reclamado' (0/1).",
            )
            if excel_eval:
                try:
                    sheets_ev = pd.ExcelFile(excel_eval).sheet_names; excel_eval.seek(0)
                    sel_ev = st.selectbox(
                        "Seleccionar hoja de evaluación", options=sheets_ev,
                        index=sheets_ev.index("Dureza (2)") if "Dureza (2)" in sheets_ev else 0,
                        key="json_eval_sheet",
                    )
                    excel_eval.seek(0)
                    df_ev = pd.read_excel(excel_eval, sheet_name=sel_ev)
                    df_ev.columns = df_ev.columns.str.strip()
                except Exception as exc:
                    st.error(f"Error al leer archivo: {exc}"); df_ev = None

                if df_ev is not None:
                    miss_ev = [c for c in DUREZA_COLS if c not in df_ev.columns]
                    if miss_ev:
                        st.error(f"Faltan columnas de dureza: {miss_ev[:4]} …")
                    elif "Reclamado" not in df_ev.columns:
                        st.error("Falta la columna **'Reclamado'** (0/1).")
                    else:
                        model_json = st.session_state.get("json_model")
                        if model_json is None and os.path.exists(JSON_MODEL_PATH):
                            model_json = XGBClassifier()
                            model_json.load_model(JSON_MODEL_PATH)
                            st.session_state["json_model"] = model_json
                        vars_ev    = df_ev.apply(calcular_variables, axis=1)
                        df_ev      = pd.concat([df_ev.reset_index(drop=True), vars_ev.reset_index(drop=True)], axis=1)
                        df_ev      = df_ev.loc[:, ~df_ev.columns.duplicated(keep="first")]
                        X_ev       = df_ev[FEATURES]
                        y_ev       = df_ev["Reclamado"].astype(int)
                        df_ev["Prob_modelo"] = model_json.predict_proba(X_ev)[:, 1]
                        df_ev["Decision"]    = df_ev.apply(
                            lambda r: clasificar(r["Prob_modelo"], r["STDEV"]), axis=1)
                        y_pred_ev  = (df_ev["Decision"] == "RECHAZAR").astype(int)
                        report_ev  = classification_report(y_ev, y_pred_ev, output_dict=True, zero_division=0)
                        cm_ev      = confusion_matrix(y_ev, y_pred_ev)
                        st.session_state["json_eval_info"] = {
                            "n_rows": len(df_ev), "cm": cm_ev.tolist(), "report": report_ev,
                            "df_scatter": df_ev[
                                ["Variabilidad", "Simetría", "Curvatura", "Prob_modelo", "Decision"]
                            ].to_dict("records"),
                        }
                        st.rerun()

            st.markdown("---")
            st.markdown("#### Cargar un modelo diferente")

        # JSON uploader (always visible)
        uploaded_json = st.file_uploader(
            "Cargar modelo pre-entrenado (.json)", type=["json"],
            key="json_model_uploader",
            help="Modelo XGBoost exportado con model.save_model('modelo.json').",
        )
        if uploaded_json:
            try:
                # Save model file permanently to disk
                model_bytes = uploaded_json.read()
                with open(JSON_MODEL_PATH, "wb") as f:
                    f.write(model_bytes)
                model_json = XGBClassifier()
                model_json.load_model(JSON_MODEL_PATH)
                try:
                    n_trees = model_json.get_booster().num_boosted_rounds()
                except Exception:
                    n_trees = "?"
                importances_j = dict(zip(FEATURES, model_json.feature_importances_.tolist()))
                json_info = {"n_trees": n_trees, "importances": importances_j}
                # Save info to disk
                _save_json(JSON_INFO_PATH, json_info)
                st.session_state["json_model"]   = model_json
                st.session_state["json_info"]    = json_info
                st.session_state["model_source"] = "JSON cargado"
                st.session_state.pop("json_eval_info", None)
                st.rerun()
            except Exception as exc:
                st.error(f"Error al cargar el modelo: {exc}")
                st.info("Asegúrese de que el archivo sea un modelo XGBoost válido en formato JSON.")
        elif st.session_state.get("json_model") is None and not os.path.exists(JSON_MODEL_PATH):
            st.markdown("""
            <div class="placeholder-card">
                <div class="placeholder-icon"><i class="bi bi-file-earmark-code"></i></div>
                <div class="placeholder-title">Suba un modelo XGBoost (.json)</div>
                <div class="placeholder-desc">
                    Exportado con <strong>model.save_model('modelo.json')</strong><br><br>
                    Una vez cargado, estará disponible en Predicción Reel y Análisis Histórico.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Access settings ───────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Configuración de acceso"):
        cfg = load_config()
        if cfg.get("admin_password"):
            st.info("El acceso a Administrador está protegido con contraseña.")
        else:
            st.warning("Sin contraseña configurada. Configure una para proteger el acceso.")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            new_pwd = st.text_input("Nueva contraseña", type="password", key="new_pwd")
        with col_p2:
            confirm_pwd = st.text_input("Confirmar contraseña", type="password", key="confirm_pwd")

        col_s, col_l, _ = st.columns([1, 1, 2])
        with col_s:
            if st.button("Guardar contraseña", key="save_pwd"):
                if not new_pwd:
                    st.error("La contraseña no puede estar vacía")
                elif new_pwd != confirm_pwd:
                    st.error("Las contraseñas no coinciden")
                else:
                    cfg["admin_password"] = new_pwd
                    save_config(cfg)
                    st.success("Contraseña actualizada")
        with col_l:
            if cfg.get("admin_password") and st.button("Cerrar sesión", key="logout"):
                st.session_state["admin_auth"] = False
                st.rerun()


# =============================================================================
# PARTE 2 — PREDICCIÓN EN TIEMPO REAL
# =============================================================================
elif "Predicción" in modo:

    st.markdown(
        '<div class="section-hdr"><h2>'
        '<i class="bi bi-graph-up-arrow"></i>'
        ' Predicción en Tiempo Real — Análisis de Bobina Reel a Reel'
        '</h2></div>',
        unsafe_allow_html=True,
    )

    model_rt = load_model()
    if model_rt is None:
        st.warning("Sin modelo activo. Vaya a **Administrador** para entrenar con Excel o cargar un modelo (.json).")
        st.stop()

    col_in, col_out = st.columns([1, 2], gap="large")

    with col_in:
        st.markdown("### Mediciones de Dureza")
        st.markdown(
            "Pegue los **20 valores** de dureza (uno por línea). "
            "Se aceptan punto o coma como decimal."
        )
        dureza_input = st.text_area(
            "Dureza", placeholder="30.00\n44.00\n42.00\n38.00\n45.00\n…\n(20 valores)",
            height=440, label_visibility="collapsed",
        )
        btn_analizar = st.button("Analizar Bobina", use_container_width=True)
        st.markdown("---")
        st.markdown("""
**Instrucciones:**
1. Ingrese los 20 valores de dureza
2. Un valor por línea (columna de Excel)
3. El análisis es automático
4. Punto **o** coma como decimal
        """)

    with col_out:
        if btn_analizar:
            if not dureza_input.strip():
                st.error("Ingrese las mediciones de dureza antes de analizar.")
            else:
                try:
                    lines = [
                        ln.strip().replace(",", ".")
                        for ln in dureza_input.strip().split("\n") if ln.strip()
                    ]
                    if len(lines) != 20:
                        st.error(f"Se necesitan exactamente **20 valores**. Se detectaron: **{len(lines)}**")
                        st.stop()

                    durezas  = [float(v) for v in lines]
                    vars_s   = calcular_variables(durezas)
                    X_pred   = pd.DataFrame([vars_s])[FEATURES]
                    prob     = float(model_rt.predict_proba(X_pred)[0, 1])
                    decision = clasificar(prob, float(vars_s["STDEV"]))

                    if decision == "RECHAZAR":
                        st.markdown("""
                        <div class="badge-rechazar">
                            <div class="badge-icon"><i class="bi bi-x-circle-fill"></i></div>
                            <div class="badge-text">RECHAZAR</div>
                            <div class="badge-sub">Alta variabilidad en la dureza — alto riesgo de bagginess.<br>
                            No enviar a corrugadora sin revisión.</div>
                        </div>""", unsafe_allow_html=True)
                    elif decision == "ALERTA":
                        st.markdown("""
                        <div class="badge-alerta">
                            <div class="badge-icon"><i class="bi bi-exclamation-triangle-fill"></i></div>
                            <div class="badge-text">ALERTA</div>
                            <div class="badge-sub">Variabilidad detectada — revisar bobina antes de despachar.<br>
                            Riesgo moderado de reclamo en corrugadora.</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="badge-ok">
                            <div class="badge-icon"><i class="bi bi-check-circle-fill"></i></div>
                            <div class="badge-text">OK</div>
                            <div class="badge-sub">Bobina dentro de parámetros normales.<br>
                            Dureza estable en todo el ancho.</div>
                        </div>""", unsafe_allow_html=True)

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Prob. Reclamo", f"{prob:.1%}")
                    m2.metric("STDEV",         f"{float(vars_s['STDEV']):.2f}")
                    m3.metric("Curvatura",     f"{float(vars_s['Curvatura']):.2f}")
                    m4.metric("Simetría",      f"{float(vars_s['Simetría']):.2f}")
                    m5.metric("Variabilidad",  f"{float(vars_s['Variabilidad']):.1f}")

                    st.markdown("### Perfil de Dureza — Ancho de Bobina")
                    arr    = np.array(durezas, dtype=float)
                    pos    = np.arange(1, 21)
                    mean_a = np.mean(arr); std_a = np.std(arr)

                    def _pt_color(v):
                        dev = abs(v - mean_a)
                        if dev > 2 * std_a: return "#cf1322"
                        if dev > std_a:     return "#fa8c16"
                        return "#27ae60"
                    pt_colors = [_pt_color(v) for v in arr]

                    fig_bob, (ax_top, ax_bot) = plt.subplots(
                        2, 1, figsize=(13.5, 7),
                        gridspec_kw={"height_ratios": [1, 5]},
                    )
                    fig_bob.patch.set_facecolor("white")
                    for i, col_ in enumerate(pt_colors):
                        ax_top.barh(0, 1, left=i, color=col_, height=1.15, edgecolor="white", linewidth=0.7)
                    for x_e in [0, 20]:
                        ax_top.add_patch(plt.Circle(
                            (x_e, 0), 0.58, color="#8899aa", zorder=4,
                            transform=ax_top.transData, clip_on=False))
                    ax_top.set_xlim(-0.6, 20.6); ax_top.set_ylim(-0.65, 0.65); ax_top.axis("off")
                    ax_top.set_title(
                        "← Borde Izquierdo  |  Representación Visual del Ancho de Bobina  |  Borde Derecho →",
                        fontsize=9.5, color="#445566", pad=6)
                    for xb in [4, 8, 12, 16]:
                        ax_top.axvline(x=xb, color="white", linewidth=1.5, alpha=0.55, ymin=0.1, ymax=0.9)

                    for i in range(len(arr) - 1):
                        ax_bot.plot([pos[i], pos[i+1]], [arr[i], arr[i+1]],
                                    color=pt_colors[i], linewidth=2.8,
                                    solid_capstyle="round", solid_joinstyle="round")
                    ax_bot.scatter(pos, arr, c=pt_colors, s=92, zorder=5, edgecolors="white", linewidths=1.2)
                    ax_bot.axhline(y=mean_a, color="#1557a0", linestyle="--",
                                   alpha=0.65, linewidth=1.6, label=f"Media: {mean_a:.1f}")
                    ax_bot.fill_between(pos, mean_a - std_a, mean_a + std_a, alpha=0.08, color="#1557a0")

                    zone_info      = [(4.5,"M5\n(Borde Izq.)"),(8.5,"M4"),(12.5,"M3\n(Centro)"),(16.5,"M1\n(Borde Der.)")]
                    y_lp           = arr.min() - (arr.max()-arr.min())*0.09
                    zone_x_centers = [2.5, 6.5, 10.5, 14.5, 18.5]
                    zone_names     = ["M5", "M4", "M3", "—", "M1"]
                    for xb, _ in zone_info:
                        ax_bot.axvline(x=xb, color="#aabbcc", linestyle=":", linewidth=1.1, alpha=0.8)
                    for xc, zn in zip(zone_x_centers, zone_names):
                        if zn != "—":
                            ax_bot.text(xc, y_lp, zn, ha="center", fontsize=8.5, color="#6680a0", fontweight="500")

                    ax_bot.set_xlabel("Punto de Medición  (1 = Borde Izquierdo → 20 = Borde Derecho)", fontsize=10.5)
                    ax_bot.set_ylabel("Dureza", fontsize=10.5)
                    ax_bot.set_xticks(pos); ax_bot.set_xticklabels([str(p) for p in pos], fontsize=8)
                    ax_bot.set_title(
                        f"Perfil de Dureza  |  STDEV: {std_a:.2f}  |  "
                        f"Variabilidad: {float(vars_s['Variabilidad']):.1f}  |  "
                        f"Curvatura: {float(vars_s['Curvatura']):.2f}",
                        fontsize=11, fontweight="bold", pad=8)
                    leg_handles = [
                        mpatches.Patch(color="#27ae60", label="Normal (±1σ)"),
                        mpatches.Patch(color="#fa8c16", label="Variación (1σ–2σ)"),
                        mpatches.Patch(color="#cf1322", label="Crítico (>2σ)"),
                        plt.Line2D([0],[0], color="#1557a0", linestyle="--", label=f"Media: {mean_a:.1f}"),
                    ]
                    ax_bot.legend(handles=leg_handles, loc="upper right", fontsize=8.5, framealpha=0.88)
                    plt.tight_layout(pad=1.6)
                    st.pyplot(fig_bob, use_container_width=True)

                    with st.expander("Resumen por zona y variables calculadas"):
                        z1, z2, z3, z4 = st.columns(4)
                        z1.metric("M5 (Borde Izq.)", f"{float(vars_s['M5']):.2f}")
                        z2.metric("M4",              f"{float(vars_s['M4']):.2f}")
                        z3.metric("M3 (Centro)",     f"{float(vars_s['M3']):.2f}")
                        z4.metric("M1 (Borde Der.)", f"{float(vars_s['M1']):.2f}")
                        st.dataframe(
                            pd.DataFrame([{k: round(float(v), 4) for k, v in vars_s.items()}]),
                            hide_index=True, use_container_width=True)

                except ValueError as ve:
                    st.error(f"Error al convertir valores: {ve}. Verifique que todos sean números.")
                except Exception as exc:
                    st.error(f"Error inesperado: {exc}")
        else:
            st.markdown("""
            <div class="placeholder-card">
                <div class="placeholder-icon"><i class="bi bi-layers-half"></i></div>
                <div class="placeholder-title">Ingrese las 20 mediciones de dureza</div>
                <div class="placeholder-desc">
                    La herramienta calculará automáticamente el riesgo de bagginess,<br>
                    el perfil de dureza del ancho de bobina y la decisión de proceso.<br><br>
                    <strong>RECHAZAR</strong> · <strong>ALERTA</strong> · <strong>OK</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PARTE 3 — ANÁLISIS HISTÓRICO
# =============================================================================
elif "Histórico" in modo:

    st.markdown(
        '<div class="section-hdr"><h2>'
        '<i class="bi bi-folder-open"></i>'
        ' Análisis Histórico — Múltiples Reeles'
        '</h2></div>',
        unsafe_allow_html=True,
    )

    model_hist = load_model()
    if model_hist is None:
        st.warning("Sin modelo activo. Vaya a **Administrador** para entrenar con Excel o cargar un modelo (.json).")
        st.stop()

    # ── Excel structure spec ──────────────────────────────────────────────────
    with st.expander("Estructura requerida del archivo Excel", expanded=False):
        st.markdown("""
<div class="spec-card">
<h4>Columnas obligatorias</h4>

| Columna | Descripción |
|---------|-------------|
| <code>Dureza Rollo (1)</code> … <code>Dureza Rollo (20)</code> | 20 valores numéricos de dureza. Punto o coma como decimal. |

<h4 style="margin-top:14px;">Columnas opcionales (enriquecen el análisis)</h4>

| Columna | Descripción |
|---------|-------------|
| <code>Date</code> | Fecha del reel — activa el resumen por fecha |
| <code>Name</code> | Identificador o nombre del reel |
| <code>Product (short)</code> | Nombre corto del producto |
| <code>Reclamado</code> | 0 = sin reclamo · 1 = reclamado (solo si se quiere comparar con la realidad) |

**Notas:**
- Cada fila representa un reel individual
- El orden de las columnas no importa, solo los nombres exactos
- La hoja a analizar se selecciona manualmente si el archivo tiene varias
</div>
        """, unsafe_allow_html=True)

    uploaded_h = st.file_uploader(
        "Cargar Excel con múltiples reeles",
        type=["xlsx", "xls"],
        help="Columnas requeridas: 'Dureza Rollo (1)' al '(20)'. Opcionales: Date, Name, Product (short).",
    )

    if uploaded_h:
        try:
            sheet_names_h = pd.ExcelFile(uploaded_h).sheet_names; uploaded_h.seek(0)
        except Exception as exc:
            st.error(f"No se pudo leer el archivo: {exc}"); st.stop()

        selected_sheet_h = st.selectbox(
            "Seleccionar hoja", options=sheet_names_h,
            index=sheet_names_h.index("Dureza (2)") if "Dureza (2)" in sheet_names_h else 0,
        )
        with st.spinner("Procesando reeles…"):
            try:
                uploaded_h.seek(0)
                df_h = pd.read_excel(uploaded_h, sheet_name=selected_sheet_h)
                df_h.columns = df_h.columns.str.strip()
            except Exception as exc:
                st.error(f"Error al leer archivo: {exc}"); st.stop()

        miss_h = [c for c in DUREZA_COLS if c not in df_h.columns]
        if miss_h:
            st.error(f"Faltan columnas de dureza: {miss_h[:4]} …"); st.stop()

        vars_h    = df_h.apply(calcular_variables, axis=1)
        df_h_base = df_h.drop(columns=[c for c in FEATURES if c in df_h.columns], errors="ignore")
        df_h      = pd.concat([df_h_base.reset_index(drop=True), vars_h.reset_index(drop=True)], axis=1)
        X_h                 = df_h[FEATURES]
        df_h["Prob_modelo"] = model_hist.predict_proba(X_h)[:, 1]
        df_h["Decision"]    = df_h.apply(lambda r: clasificar(r["Prob_modelo"], r["STDEV"]), axis=1)

        st.success(f"**{len(df_h)}** reeles procesados correctamente")

        n_tot_h = len(df_h)
        n_rej_h = int((df_h["Decision"] == "RECHAZAR").sum())
        n_alt_h = int((df_h["Decision"] == "ALERTA").sum())
        n_ok_h  = int((df_h["Decision"] == "OK").sum())

        show_cols_h = [c for c in [
            "Date", "Name", "Product (short)", "Decision",
            "Prob_modelo", "STDEV", "Variabilidad", "Curvatura", "Simetría",
        ] if c in df_h.columns]

        # ── Buscar Reel ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Buscar Reel")
        if "Name" in df_h.columns:
            col_srch, col_res = st.columns([1, 2])
            with col_srch:
                reel_query = st.text_input("Número / nombre del reel", placeholder="Ej: R-2024-001",
                                           key="hist_reel_search")
            if reel_query:
                match_h = df_h[df_h["Name"].astype(str).str.strip().str.lower()
                                == reel_query.strip().lower()]
                with col_res:
                    if len(match_h) == 0:
                        st.warning(f"No se encontró el reel '{reel_query}'")
                    else:
                        row_h = match_h.iloc[0]
                        dec_h = row_h["Decision"]
                        ico_h = {"RECHAZAR": "🔴 RECHAZAR", "ALERTA": "🟡 ALERTA", "OK": "🟢 OK"}.get(dec_h, dec_h)
                        st.markdown(
                            f"<div style='padding:12px 16px;border-radius:10px;"
                            f"background:#1e2130;border:1px solid #333;margin-top:4px'>"
                            f"<b style='font-size:1.05rem'>{ico_h}</b><br>"
                            f"<span style='color:#aaa'>STDEV:</span> <b>{row_h['STDEV']:.3f}</b>"
                            f"&nbsp;&nbsp;"
                            f"<span style='color:#aaa'>Probabilidad:</span> <b>{row_h['Prob_modelo']:.1%}</b>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
        else:
            st.info("Añada la columna **Name** al Excel para habilitar la búsqueda por reel.")

        # ── KPI cards ─────────────────────────────────────────────────────────
        st.markdown("---")
        pct_rej = f"{n_rej_h/n_tot_h:.0%}" if n_tot_h else "0%"
        pct_alt = f"{n_alt_h/n_tot_h:.0%}" if n_tot_h else "0%"
        pct_ok  = f"{n_ok_h/n_tot_h:.0%}"  if n_tot_h else "0%"
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(
            "<div style='padding:16px 20px;border-radius:12px;background:#05B1FF'>"
            "<div style='font-size:0.82rem;color:#fff;opacity:0.85;margin-bottom:4px'>Total Reeles</div>"
            f"<div style='font-size:1.9rem;font-weight:700;color:#fff'>{n_tot_h}</div>"
            "</div>", unsafe_allow_html=True)
        k2.markdown(
            "<div style='padding:16px 20px;border-radius:12px;background:#FF0000'>"
            "<div style='font-size:0.82rem;color:#fff;opacity:0.85;margin-bottom:4px'>Rechazados</div>"
            f"<div style='font-size:1.9rem;font-weight:700;color:#fff'>"
            f"{n_rej_h} <span style='font-size:1rem;font-weight:400;opacity:0.85'>({pct_rej})</span>"
            f"</div></div>", unsafe_allow_html=True)
        k3.markdown(
            "<div style='padding:16px 20px;border-radius:12px;background:#FFC000'>"
            "<div style='font-size:0.82rem;color:#fff;opacity:0.9;margin-bottom:4px'>En Alerta</div>"
            f"<div style='font-size:1.9rem;font-weight:700;color:#fff'>"
            f"{n_alt_h} <span style='font-size:1rem;font-weight:400;opacity:0.85'>({pct_alt})</span>"
            f"</div></div>", unsafe_allow_html=True)
        k4.markdown(
            "<div style='padding:16px 20px;border-radius:12px;background:#33CC33'>"
            "<div style='font-size:0.82rem;color:#fff;opacity:0.85;margin-bottom:4px'>OK</div>"
            f"<div style='font-size:1.9rem;font-weight:700;color:#fff'>"
            f"{n_ok_h} <span style='font-size:1rem;font-weight:400;opacity:0.85'>({pct_ok})</span>"
            f"</div></div>", unsafe_allow_html=True)

        has_date = "Date" in df_h.columns
        resumen = None
        analisis_df = None

        if has_date:
            df_h["_fecha"] = pd.to_datetime(df_h["Date"], errors="coerce").dt.date
            resumen = df_h.groupby("_fecha").agg(
                Total_Reeles      = ("Decision",     "count"),
                Rechazados        = ("Decision",     lambda x: (x == "RECHAZAR").sum()),
                Alerta            = ("Decision",     lambda x: (x == "ALERTA").sum()),
                OK                = ("Decision",     lambda x: (x == "OK").sum()),
                STDEV_promedio    = ("STDEV",        lambda x: round(x.mean(), 2)),
                Variabilidad_prom = ("Variabilidad", lambda x: round(x.mean(), 1)),
                Curvatura_prom    = ("Curvatura",    lambda x: round(x.mean(), 2)),
            ).reset_index().rename(columns={"_fecha": "Fecha"})
            fechas_str = [str(f) for f in resumen["Fecha"]]
            n_dias = len(fechas_str)

            # ── Gráfica de barras apiladas por día ────────────────────────────
            st.markdown("---")
            fig_s, ax_s = plt.subplots(figsize=(max(8, n_dias * 1.1), 6))
            fig_s.patch.set_facecolor("white")
            ax_s.set_facecolor("white")

            b_ok  = ax_s.bar(fechas_str, resumen["OK"],
                             label="OK",         color="#33CC33", width=0.55)
            b_alt = ax_s.bar(fechas_str, resumen["Alerta"],
                             label="En Alerta",  color="#FFC000", width=0.55,
                             bottom=resumen["OK"])
            b_rej = ax_s.bar(fechas_str, resumen["Rechazados"],
                             label="Rechazados", color="#FF0000", width=0.55,
                             bottom=resumen["OK"] + resumen["Alerta"])

            for bar, v in zip(b_ok, resumen["OK"]):
                if v > 0:
                    ax_s.text(bar.get_x() + bar.get_width() / 2,
                              bar.get_y() + v / 2,
                              str(v), ha="center", va="center",
                              fontsize=16, color="white", fontweight="bold")
            for bar, v, base in zip(b_alt, resumen["Alerta"], resumen["OK"]):
                if v > 0:
                    ax_s.text(bar.get_x() + bar.get_width() / 2,
                              base + v / 2,
                              str(v), ha="center", va="center",
                              fontsize=16, color="#333", fontweight="bold")
            for bar, v, base in zip(b_rej, resumen["Rechazados"],
                                    resumen["OK"] + resumen["Alerta"]):
                if v > 0:
                    ax_s.text(bar.get_x() + bar.get_width() / 2,
                              base + v / 2,
                              str(v), ha="center", va="center",
                              fontsize=16, color="white", fontweight="bold")

            ax_s.set_xlabel("Fecha", color="#111", fontsize=16)
            ax_s.set_ylabel("Cantidad de Reeles", color="#111", fontsize=16)
            ax_s.set_title("Reeles por Día", color="#111", fontsize=20, fontweight="bold")
            ax_s.tick_params(colors="#111", labelsize=14)
            ax_s.spines[["top", "right"]].set_visible(False)
            ax_s.spines[["left", "bottom"]].set_color("#bbb")
            ax_s.legend(facecolor="white", labelcolor="#111", edgecolor="#bbb", fontsize=15)
            plt.xticks(rotation=45, ha="right", color="#111")
            plt.tight_layout()
            st.pyplot(fig_s)
            plt.close(fig_s)

            # ── Resumen por categoría ─────────────────────────────────────────
            st.markdown("---")
            st.markdown("### Resumen por Categoría")

            cat_sel = st.radio("Seleccionar categoría",
                               ["OK", "En Alerta", "Rechazado"],
                               horizontal=True, key="cat_radio_hist")

            cat_col_map  = {"OK": "OK", "En Alerta": "Alerta", "Rechazado": "Rechazados"}
            cat_dec_map  = {"OK": "OK", "En Alerta": "ALERTA",  "Rechazado": "RECHAZAR"}
            cat_clr_map  = {"OK": "#33CC33", "En Alerta": "#FFC000", "Rechazado": "#FF0000"}
            col_data = cat_col_map[cat_sel]
            col_dec  = cat_dec_map[cat_sel]
            col_clr  = cat_clr_map[cat_sel]

            fig_c, ax_c = plt.subplots(figsize=(max(10, n_dias * 1.3), 6))
            fig_c.patch.set_facecolor("white")
            ax_c.set_facecolor("white")
            b_cat = ax_c.bar(fechas_str, resumen[col_data], color=col_clr, width=0.6)
            for b, v in zip(b_cat, resumen[col_data]):
                if v > 0:
                    ax_c.text(b.get_x() + b.get_width() / 2,
                              b.get_height() + 0.05,
                              str(v), ha="center", va="bottom",
                              fontsize=18, fontweight="bold", color="#111")
            ax_c.set_xlabel("Fecha", color="#111", fontsize=16)
            ax_c.set_ylabel("Cantidad de Reeles", color="#111", fontsize=16)
            ax_c.set_title(f"Reeles {cat_sel} por Día", color="#111",
                           fontsize=20, fontweight="bold")
            ax_c.tick_params(colors="#111", labelsize=14)
            ax_c.spines[["top", "right"]].set_visible(False)
            ax_c.spines[["left", "bottom"]].set_color("#bbb")
            plt.xticks(rotation=45, ha="right", color="#111")
            plt.tight_layout()
            st.pyplot(fig_c)
            plt.close(fig_c)

            with st.expander(f"Detalles — {cat_sel}", expanded=False):
                df_cat = df_h[df_h["Decision"] == col_dec][show_cols_h].copy()
                if "Prob_modelo" in df_cat.columns:
                    df_cat["Prob_modelo"] = df_cat["Prob_modelo"].map(lambda x: f"{x:.1%}")
                if len(df_cat):
                    st.dataframe(df_cat.round(2), use_container_width=True, hide_index=True)
                else:
                    st.success(f"No hay reeles en la categoría '{cat_sel}'")

            # ── Resumen por fecha (desplegable) ───────────────────────────────
            st.markdown("---")
            with st.expander("Resumen por Fecha", expanded=False):
                st.dataframe(resumen, use_container_width=True, hide_index=True)

            # ── Análisis predictivo diario (desplegable) ──────────────────────
            st.markdown("---")
            with st.expander("Análisis Predictivo Diario", expanded=False):
                def _analisis_diario(g):
                    msgs = []
                    if g["STDEV"].mean() > 3:
                        msgs.append("Alta variabilidad en dureza — posible riesgo de bagginess")
                    if g["Variabilidad"].mean() > 10:
                        msgs.append("Amplio rango de dureza en el ancho de bobina")
                    if g["Curvatura"].mean() > 3:
                        msgs.append("Posible problema de perfil transversal")
                    return " | ".join(msgs) if msgs else "Condiciones estables"

                analisis_df = (df_h.groupby("_fecha")
                               .apply(_analisis_diario)
                               .reset_index())
                analisis_df.columns = ["Fecha", "Análisis Predictivo"]
                st.dataframe(analisis_df, use_container_width=True, hide_index=True)

        else:
            # Sin columna Date — vista simplificada
            # ── Resumen por categoría (sin fechas) ───────────────────────────
            st.markdown("---")
            st.markdown("### Resumen por Categoría")
            st.info("Añada la columna **Date** al Excel para ver las gráficas temporales.")
            col_sel_nd = st.radio("Seleccionar categoría",
                                  ["OK", "En Alerta", "Rechazado"],
                                  horizontal=True, key="cat_radio_nd")
            cat_dec_nd = {"OK": "OK", "En Alerta": "ALERTA", "Rechazado": "RECHAZAR"}
            with st.expander(f"Detalles — {col_sel_nd}", expanded=False):
                df_cat_nd = df_h[df_h["Decision"] == cat_dec_nd[col_sel_nd]][show_cols_h].copy()
                if "Prob_modelo" in df_cat_nd.columns:
                    df_cat_nd["Prob_modelo"] = df_cat_nd["Prob_modelo"].map(lambda x: f"{x:.1%}")
                if len(df_cat_nd):
                    st.dataframe(df_cat_nd.round(2), use_container_width=True, hide_index=True)
                else:
                    st.success(f"No hay reeles en la categoría '{col_sel_nd}'")

            st.markdown("---")
            with st.expander("Análisis Predictivo Diario", expanded=False):
                msgs_nd = []
                if df_h["STDEV"].mean() > 3:
                    msgs_nd.append("Alta variabilidad en dureza — posible riesgo de bagginess")
                if df_h["Variabilidad"].mean() > 10:
                    msgs_nd.append("Amplio rango de dureza en el ancho de bobina")
                if df_h["Curvatura"].mean() > 3:
                    msgs_nd.append("Posible problema de perfil transversal")
                if msgs_nd:
                    st.warning(" | ".join(msgs_nd))
                else:
                    st.success("Condiciones estables")

        # ── Exportar ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Exportar Resultados")
        buf_h = io.BytesIO()
        with pd.ExcelWriter(buf_h, engine="openpyxl") as writer:
            df_h.drop(columns=["_fecha"], errors="ignore").round(3).to_excel(
                writer, sheet_name="Detalle", index=False)
            if resumen is not None:
                resumen.to_excel(writer, sheet_name="Resumen Fecha", index=False)
            df_h[df_h["Decision"] == "RECHAZAR"][show_cols_h].round(3).to_excel(
                writer, sheet_name="Rechazados", index=False)
            df_h[df_h["Decision"] == "ALERTA"][show_cols_h].round(3).to_excel(
                writer, sheet_name="Alertas", index=False)
            if analisis_df is not None:
                analisis_df.to_excel(writer, sheet_name="Analisis", index=False)
        buf_h.seek(0)
        st.download_button(
            "Descargar resultado_final.xlsx",
            data=buf_h, file_name="resultado_final.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
