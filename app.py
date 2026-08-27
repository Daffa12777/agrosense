import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

st.set_page_config(page_title="AgroSense LoRa-X", layout="centered",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
    --orange: #f26419; --orange-soft: #f79256; --teal: #2a9d8f;
    --bg: #ffffff; --card: #fbf7f3; --border: #ececec; --text: #1a1a1a; --muted: #6b6b6b;
    --warn-bg: #fff4e6; --warn-border: #f7c59f;
}
.stApp {background: var(--bg);}
html, body, [class*="css"], .stApp, input, button, select, textarea {
    font-family: 'Inter', sans-serif !important; color: var(--text);
}
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2.5rem; max-width: 780px;}
h1 {font-weight: 800; letter-spacing: -0.02em; color: var(--text);}
h2, h3 {color: var(--text); font-weight: 700;}
p, span, label, .stMarkdown {color: var(--text) !important;}
.stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color: var(--muted) !important;}
.accent {height:4px; width:64px; background:var(--orange); border-radius:3px; margin:10px 0 4px 0;}
hr {border-color: var(--border) !important;}
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #ffffff !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
}
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {color: var(--muted) !important; font-weight:500;}
div[data-testid="stNumberInput"] button {background: var(--card) !important; color: var(--text) !important;}
div[data-testid="stMetric"] {
    background: var(--card); border: 1px solid var(--border);
    border-left: 4px solid var(--orange); border-radius: 14px; padding: 20px;
}
div[data-testid="stMetric"] label {color: var(--orange) !important; font-weight: 600;}
div[data-testid="stMetricValue"] {color: var(--text) !important; font-weight: 800;}
div[data-testid="stMetricDelta"] {color: var(--muted) !important;}
div[data-testid="stMetricDelta"] svg {display:none;}
.stButton>button {
    width: 100%; border-radius: 10px; font-weight: 700; padding: 0.6rem;
    background: var(--orange); color: #ffffff; border: none; font-size: 1rem;
}
.stButton>button:hover {background: var(--orange-soft); color:#ffffff;}
.saran-box {background: var(--card); border:1px solid var(--border);
    border-left:4px solid var(--teal); border-radius:12px; padding:16px 18px; margin:6px 0 14px 0;}
.saran-box p {margin:0; font-size:0.95rem; line-height:1.5;}
.dosis-tag {display:inline-block; background:var(--orange); color:#fff; font-weight:600;
    font-size:0.82rem; padding:3px 10px; border-radius:6px; margin-top:8px;}
.warn-item {background:var(--warn-bg); border:1px solid var(--warn-border);
    border-radius:8px; padding:9px 12px; margin:6px 0; font-size:0.86rem; color:#8a4b1e;}
.warn-item span {color:#8a4b1e !important;}
.alt-row {margin:12px 0;}
.alt-label {font-size:0.88rem; color:var(--text); margin-bottom:4px; font-weight:500;}
.alt-track {height:14px; background:#f0ede9; border-radius:5px; overflow:hidden;}
.alt-fill {height:100%; border-radius:5px;}
.alt-1 {background:var(--orange);}
.alt-2 {background:var(--orange-soft);}
.alt-3 {background:#c9c2ba;}
.xai-row {display:flex; align-items:center; gap:10px; margin:7px 0;}
.xai-name {width:150px; font-size:0.85rem; color:var(--text);}
.xai-bar {flex:1; height:18px; background:#f0ede9; border-radius:5px; overflow:hidden;}
.xai-fill-a {height:100%; background:var(--orange); border-radius:5px;}
.xai-fill-b {height:100%; background:var(--teal); border-radius:5px;}
.xai-pct {width:48px; text-align:right; font-size:0.82rem; font-weight:600;}
.xai-pct-a {color:var(--orange);}
.xai-pct-b {color:var(--teal);}
.result-note {color:var(--muted); font-size:0.82rem; margin-top:1.2rem;
    border-top:1px solid var(--border); padding-top:0.8rem;}
</style>
""", unsafe_allow_html=True)

LABELS = {
    "N_ppm":"Nitrogen (ppm)","P_ppm":"Fosfor (ppm)","K_ppm":"Kalium (ppm)",
    "soil_moisture":"Kelembapan Tanah (%)","soil_ph":"pH Tanah",
    "temperature":"Suhu (C)","humidity":"Kelembapan Udara (%)","rainfall":"Curah Hujan (mm)",
}
FEAT_ID = {
    "N_ppm":"Nitrogen","P_ppm":"Fosfor","K_ppm":"Kalium",
    "soil_moisture":"Kelembapan Tanah","soil_ph":"pH Tanah","temperature":"Suhu",
    "humidity":"Kelembapan Udara","rainfall":"Curah Hujan",
    "soil_type":"Jenis Tanah","crop":"Tanaman",
}
FERT_LABEL = {"None":"Tidak perlu pupuk","Organik":"Pupuk Organik","Kapur-Dolomit":"Kapur Dolomit"}
def nice_fert(name): return FERT_LABEL.get(name, name)

# dosis perkiraan (kg/ha) - acuan kasar, WAJIB diverifikasi penyuluh
DOSIS = {"Urea":200,"SP-36":150,"KCl":100,"NPK-16-16-16":300,
         "Kapur-Dolomit":1500,"Organik":2000,"None":0}
LUAS_HA = 0.25
LUAS_M2 = 2500

@st.cache_resource
def load_all():
    meta = json.load(open("models/meta.json"))
    pre_f = joblib.load("models/pre_fert.joblib")
    pre_i = joblib.load("models/pre_irr.joblib")
    le    = joblib.load("models/label_encoder.joblib")
    tab_f = TabNetClassifier(); tab_f.load_model("models/tabnet_fert.zip")
    tab_i = TabNetRegressor();  tab_i.load_model("models/tabnet_irr.zip")
    return meta, pre_f, pre_i, le, tab_f, tab_i

meta, pre_f, pre_i, le, tab_f, tab_i = load_all()
NUM, CAT = meta["NUM"], meta["CAT"]
to_dense = lambda a: np.asarray(a.todense() if hasattr(a, "todense") else a, dtype=np.float32)

def _orig(fn):
    if fn.startswith("num__"): return fn[5:]
    if fn.startswith("cat__"):
        for c in CAT:
            if fn[5:].startswith(c + "_"): return c
    return fn

def explain(model, pre, Xm):
    feat = list(pre.get_feature_names_out())
    groups = [_orig(f) for f in feat]
    M, _ = model.explain(Xm)
    imp = np.abs(M[0])
    agg = {}
    for g, v in zip(groups, imp): agg[g] = agg.get(g, 0.0) + float(v)
    tot = sum(agg.values()) or 1.0
    agg = {k: v / tot for k, v in agg.items()}
    return sorted(agg.items(), key=lambda x: -x[1])[:5]

def render_xai(pairs, fill_cls, pct_cls):
    for name, frac in pairs:
        st.markdown(
            f'<div class="xai-row"><div class="xai-name">{FEAT_ID.get(name,name)}</div>'
            f'<div class="xai-bar"><div class="{fill_cls}" style="width:{frac*100:.0f}%"></div></div>'
            f'<div class="xai-pct {pct_cls}">{frac*100:.0f}%</div></div>',
            unsafe_allow_html=True)

def build_advice(vals, fert, mm):
    ph = vals.get("soil_ph"); m = vals.get("soil_moisture")
    n = vals.get("N_ppm"); p = vals.get("P_ppm"); k = vals.get("K_ppm"); t = vals.get("temperature")
    warns = []
    if ph is not None and ph < 5.0:
        warns.append("pH tanah asam (di bawah 5.0). Lakukan pengapuran lebih dulu sebelum pupuk lain agar hara terserap optimal.")
    elif ph is not None and ph > 7.8:
        warns.append("pH tanah terlalu basa (di atas 7.8). Sebagian hara sulit terserap; pertimbangkan penambahan bahan organik.")
    if m is not None and m < 25:
        warns.append("Kelembapan tanah sangat rendah (di bawah 25%). Tanaman berisiko stres kering.")
    elif m is not None and m > 80:
        warns.append("Kelembapan tanah sangat tinggi (di atas 80%). Hati-hati genangan dan busuk akar; kurangi penyiraman.")
    if t is not None and t > 36:
        warns.append("Suhu sangat tinggi (di atas 36 C). Lakukan penyiraman pagi atau sore, hindari siang hari.")
    if n is not None and n < 35: warns.append("Nitrogen rendah. Pertumbuhan daun dapat terhambat.")
    if p is not None and p < 20: warns.append("Fosfor rendah. Perakaran dan pembungaan dapat terganggu.")
    if k is not None and k < 25: warns.append("Kalium rendah. Ketahanan tanaman terhadap penyakit menurun.")

    dos = DOSIS.get(fert, 0)
    dosis_txt = None
    if dos > 0:
        dosis_txt = f"Perkiraan dosis: {dos} kg/ha (sekitar {dos*LUAS_HA:.0f} kg untuk lahan 0,25 ha)"

    liter = mm * LUAS_M2
    if mm < 1:
        irr_txt = f"Kebutuhan air rendah ({mm:.2f} mm). Tanah masih cukup lembap, penyiraman dapat ditunda."
    else:
        irr_txt = f"Siram sekitar {mm:.2f} mm (kurang lebih {liter:.0f} liter untuk 0,25 ha), sebaiknya pagi hari."

    if fert == "None":
        narasi = f"Kondisi hara tanah tercukupi, tidak perlu pemupukan saat ini. {irr_txt}"
    else:
        narasi = f"Disarankan pemberian {nice_fert(fert)}. {irr_txt}"
    return narasi, dosis_txt, warns

def recommend(payload):
    row = pd.DataFrame([{**{c: np.nan for c in NUM + CAT}, **payload}])[NUM + CAT]
    Xf = to_dense(pre_f.transform(row))
    Xi = to_dense(pre_i.transform(row))
    proba = tab_f.predict_proba(Xf)[0]
    j = int(proba.argmax())
    mm = max(float(tab_i.predict(Xi).ravel()[0]), 0.0)
    fert = le.classes_[j]
    top3 = [(le.classes_[q], float(proba[q])) for q in proba.argsort()[::-1][:3]]
    xai_f = explain(tab_f, pre_f, Xf)
    xai_i = explain(tab_i, pre_i, Xi)
    narasi, dosis_txt, warns = build_advice(payload, fert, mm)
    return fert, float(proba[j]), mm, top3, xai_f, xai_i, narasi, dosis_txt, warns

st.title("AgroSense LoRa-X")
st.markdown('<div class="accent"></div>', unsafe_allow_html=True)
st.write("Rekomendasi pemupukan dan irigasi berbasis data sensor tanah dan lingkungan.")
st.divider()

st.subheader("Data Sensor")
vals = {}
cols = st.columns(2)
for i, col in enumerate(NUM):
    lo, hi, med = meta["num_ranges"][col]
    with cols[i % 2]:
        vals[col] = st.number_input(LABELS.get(col, col), min_value=float(lo),
                                    max_value=float(hi), value=float(med), step=0.1)
c1, c2 = st.columns(2)
vals["soil_type"] = c1.selectbox("Jenis Tanah", meta["soil_types"])
vals["crop"]      = c2.selectbox("Tanaman", meta["crops"])

st.write("")
if st.button("Dapatkan Rekomendasi"):
    fert, conf, mm, top3, xai_f, xai_i, narasi, dosis_txt, warns = recommend(vals)
    st.divider()

    m1, m2 = st.columns(2)
    m1.metric("Rekomendasi Pupuk", nice_fert(fert), f"{conf*100:.0f}% keyakinan")
    m2.metric("Kebutuhan Irigasi", f"{mm:.2f} mm/hari", f"~{mm*2500:.0f} L untuk 0,25 ha")

    st.write("")
    st.markdown("### Rekomendasi Tindakan")
    dosis_html = f'<div class="dosis-tag">{dosis_txt}</div>' if dosis_txt else ""
    st.markdown(f'<div class="saran-box"><p>{narasi}</p>{dosis_html}</div>',
                unsafe_allow_html=True)

    if warns:
        st.markdown("### Peringatan Kondisi Lahan")
        for w in warns:
            st.markdown(f'<div class="warn-item"><span>{w}</span></div>', unsafe_allow_html=True)

    st.write("")
    st.write("**Alternatif pupuk:**")
    for idx, (name, p) in enumerate(top3, 1):
        st.markdown(
            f'<div class="alt-row"><div class="alt-label">{nice_fert(name)} — {p*100:.0f}%</div>'
            f'<div class="alt-track"><div class="alt-fill alt-{idx}" style="width:{p*100:.0f}%"></div></div></div>',
            unsafe_allow_html=True)

    st.write("")
    st.markdown("### Penjelasan (Explainable AI)")
    st.caption("Fitur yang paling memengaruhi rekomendasi ini.")
    xc1, xc2 = st.columns(2)
    with xc1:
        st.markdown("**Faktor Pupuk**")
        render_xai(xai_f, "xai-fill-a", "xai-pct-a")
    with xc2:
        st.markdown("**Faktor Irigasi**")
        render_xai(xai_i, "xai-fill-b", "xai-pct-b")

    st.markdown('<p class="result-note">Dosis bersifat perkiraan dan wajib disesuaikan dengan '
                'rekomendasi penyuluh setempat. Model dilatih pada data sintetis untuk validasi '
                'pipeline; angka bukan hasil pengukuran lapangan.</p>', unsafe_allow_html=True)