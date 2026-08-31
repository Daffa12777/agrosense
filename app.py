import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

st.set_page_config(page_title="AgroSense LoRa-X", layout="centered",
                   initial_sidebar_state="collapsed")

# =========================================================
#  STYLE  (cream / putih / hijau tua). CSS keyframes only.
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,ital,wght@9..144,0,500;9..144,0,600;9..144,0,700;9..144,1,500;9..144,1,600&family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --cream:#f4efe3; --cream2:#eae3d2; --white:#ffffff;
  --green:#123529; --green-deep:#0e2a20; --green2:#2f6b4f; --sage:#7a9a86;
  --text:#182219; --muted:#6b7268; --muted2:#9aa093;
  --line:#e3dcc9; --line-soft:#efe9db;
  --warn-bg:#efe8d6; --warn-line:#d9c9a2; --warn:#6f5622;
  --eo:cubic-bezier(.23,1,.32,1); --eio:cubic-bezier(.77,0,.175,1);
}
.stApp{background:var(--cream);}
html,body,[class*="css"],.stApp,input,button,select,textarea{
  font-family:'Inter',system-ui,sans-serif !important; color:var(--text);
}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden; height:0;}
::selection{background:var(--green2); color:#fff;}
.block-container{
  padding-top:1.3rem; padding-bottom:3.4rem; max-width:840px;
  animation:pageUp .8s var(--eo) both;
}
@keyframes pageUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
h1,h2,h3,h4{color:var(--text); letter-spacing:-.02em;}
p,span,label,li,.stMarkdown{color:var(--text) !important;}
.stCaption,[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{color:var(--muted) !important;}
hr{border:none; border-top:1px solid var(--line) !important; margin:1.6rem 0 .4rem;}

/* ---------- HERO ---------- */
.hero{
  position:relative; overflow:hidden; border-radius:28px;
  background:radial-gradient(130% 150% at 88% 15%, #205a42 0%, #16412f 38%, var(--green) 62%, var(--green-deep) 100%);
  color:#fff; padding:58px 46px 60px; margin:4px 0 34px;
  box-shadow:0 34px 74px -46px rgba(18,53,41,.75);
  animation:heroIn 1s var(--eo) both;
}
@keyframes heroIn{from{opacity:0;transform:translateY(20px) scale(.99)}to{opacity:1;transform:none}}
.hero .glow{position:absolute; top:-30%; right:-6%; width:60%; height:150%;
  background:radial-gradient(circle, rgba(122,154,134,.28), transparent 60%); filter:blur(14px); pointer-events:none;}
.hero .topo{
  position:absolute; inset:0; pointer-events:none; opacity:.55;
  background:repeating-radial-gradient(circle at 84% 44%, rgba(255,255,255,.06) 0 1px, transparent 1px 34px);
  -webkit-mask-image:radial-gradient(circle at 84% 44%, #000 0%, transparent 72%);
          mask-image:radial-gradient(circle at 84% 44%, #000 0%, transparent 72%);
}
/* emitting signal rings */
.hero .rings{position:absolute; top:50%; right:9%; transform:translateY(-50%); width:330px; height:330px; pointer-events:none;}
.hero .rings s{position:absolute; inset:0; margin:auto; border-radius:50%; border:1px solid rgba(160,200,178,.4); animation:emit 4.8s linear infinite; display:block;}
.hero .rings s:nth-child(2){animation-delay:1.2s}
.hero .rings s:nth-child(3){animation-delay:2.4s}
.hero .rings s:nth-child(4){animation-delay:3.6s}
.hero .rings b{position:absolute; inset:0; margin:auto; width:11px; height:11px; border-radius:50%; background:#bfe0cf; box-shadow:0 0 28px 8px rgba(160,200,178,.6);}
@keyframes emit{0%{width:9%;height:9%;opacity:0}9%{opacity:.85}100%{width:100%;height:100%;opacity:0}}
@media(max-width:680px){.hero .rings{display:none}}

.hero .eyebrow{position:relative; display:inline-flex; align-items:center; gap:10px; font-size:.8rem; font-weight:600; color:rgba(214,228,220,.9) !important; margin-bottom:22px; letter-spacing:.01em;}
.hero .eyebrow .bar{width:26px; height:1px; background:var(--sage);}
.hero h1{position:relative; font-family:'Fraunces',serif; font-weight:600; font-size:3.05rem; line-height:1.03; color:#fff !important; letter-spacing:-.015em;}
.hero h1 .soft{color:rgba(191,224,207,.85) !important; font-style:italic; font-weight:500;}
.hero .lead{position:relative; margin-top:22px; max-width:42ch; font-size:1.05rem; line-height:1.6; color:rgba(233,240,234,.82) !important;}
.hbadges{position:relative; display:flex; gap:9px; margin-top:30px; flex-wrap:wrap;}
.hbadge{background:rgba(255,255,255,.09); border:1px solid rgba(255,255,255,.18); color:rgba(240,246,242,.92) !important;
  font-size:.78rem; font-weight:600; padding:8px 15px; border-radius:999px; backdrop-filter:blur(4px);}
@media(max-width:640px){.hero{padding:42px 26px 46px} .hero h1{font-size:2.2rem}}

/* ---------- section labels ---------- */
.kick{display:inline-flex; align-items:center; gap:10px; color:var(--green2) !important; font-size:.82rem; font-weight:600; margin:8px 0 2px;}
.kick .bar{width:22px; height:1px; background:var(--green2);}
.sub{color:var(--muted) !important; font-size:.98rem; margin:0 0 8px;}

/* ---------- INPUTS ---------- */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"]>div{
  background:var(--white) !important; color:var(--text) !important;
  border:1px solid var(--line) !important; border-radius:12px !important;
  transition:border-color .2s ease, box-shadow .2s ease;
}
div[data-testid="stNumberInput"] input:focus{
  border-color:var(--green2) !important; box-shadow:0 0 0 4px rgba(47,107,79,.14) !important;
}
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label{color:var(--muted) !important; font-weight:500;}
div[data-testid="stNumberInput"] button{background:var(--cream2) !important; color:var(--text) !important; border-color:var(--line) !important;}

/* ---------- BUTTON  (fix kontras teks) ---------- */
.stButton>button{
  width:100%; border-radius:14px; font-weight:700; padding:.9rem; font-size:1rem;
  background:var(--green) !important; border:none !important;
  transition:transform .16s var(--eo), background .25s ease, box-shadow .25s ease;
  box-shadow:0 14px 30px -18px rgba(18,53,41,.8);
}
.stButton>button, .stButton>button *,
.stButton>button p, .stButton>button div, .stButton>button span{color:#ffffff !important;}
.stButton>button:hover{background:var(--green2) !important;}
.stButton>button:active{transform:scale(.985);}
.stButton>button:focus{box-shadow:0 0 0 4px rgba(47,107,79,.25) !important;}

/* ---------- RESULT HEADER + CHIPS ---------- */
.res-head{display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px; margin:6px 0 18px; animation:rise .55s var(--eo) both;}
.res-title{font-family:'Fraunces',serif; font-size:1.5rem; font-weight:600; letter-spacing:-.01em;}
.chips{display:flex; gap:8px; flex-wrap:wrap;}
.chip{background:var(--white); border:1px solid var(--line); border-radius:999px; padding:6px 13px; font-size:.8rem; color:var(--green2) !important; font-weight:600;}

/* ---------- METRIC CARDS ---------- */
.metrics{display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:6px;}
@media(max-width:600px){.metrics{grid-template-columns:1fr}}
.metric{
  position:relative; background:var(--white); border:1px solid var(--line);
  border-radius:20px; padding:22px 24px; display:flex; align-items:center; gap:16px;
  box-shadow:0 20px 44px -34px rgba(18,53,41,.4);
  animation:rise .6s var(--eo) both;
}
.metric:nth-child(2){animation-delay:.08s;}
.metric .mc{flex:1; min-width:0;}
.metric .k{font-size:.76rem; font-weight:600; color:var(--green2) !important; letter-spacing:.02em;}
.metric .v{font-family:'Fraunces',serif; font-size:1.7rem; font-weight:600; letter-spacing:-.01em; margin-top:8px; line-height:1.05; color:var(--text) !important;}
.metric .v .unit{font-family:'Inter'; font-size:.92rem; font-weight:500; color:var(--muted) !important;}
.metric .s{font-size:.83rem; color:var(--muted) !important; margin-top:8px;}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}

/* confidence donut */
.donut{position:relative; width:76px; height:76px; flex-shrink:0; border-radius:50%;
  background:conic-gradient(var(--green2) calc(var(--p)*1%), var(--cream2) 0);
  display:grid; place-items:center; animation:pop .7s var(--eo) both;}
.donut::after{content:""; position:absolute; width:57px; height:57px; border-radius:50%; background:var(--white);}
.donut .dl{position:relative; z-index:1; font-family:'Fraunces',serif; font-size:1.02rem; font-weight:600; color:var(--green) !important;}
@keyframes pop{from{opacity:0;transform:scale(.7)}to{opacity:1;transform:scale(1)}}

/* water level chip inside irrigation card */
.wlevel{width:76px; height:76px; flex-shrink:0; border-radius:18px; overflow:hidden; position:relative; background:var(--cream); border:1px solid var(--line); animation:pop .7s var(--eo) .05s both;}
.wlevel .fillw{position:absolute; left:0; right:0; bottom:0; height:var(--w); background:linear-gradient(180deg,var(--sage),var(--green2)); animation:growH .9s var(--eo);}
@keyframes growH{from{height:0}to{height:var(--w)}}

/* ---------- blocks ---------- */
.blk{font-family:'Fraunces',serif; font-size:1.12rem; font-weight:600; margin:30px 0 13px; letter-spacing:-.01em;}
.advice{background:var(--white); border:1px solid var(--line); border-left:4px solid var(--green2); border-radius:16px; padding:18px 20px; animation:rise .6s var(--eo) .05s both;}
.advice p{margin:0; font-size:.98rem; line-height:1.65;}
.dose{display:inline-block; margin-top:12px; background:var(--green); color:#fff !important; font-weight:600; font-size:.82rem; padding:6px 13px; border-radius:9px;}

.warn{background:var(--warn-bg); border:1px solid var(--warn-line); border-radius:12px; padding:12px 15px; margin:8px 0; font-size:.9rem; color:var(--warn) !important; line-height:1.55; display:flex; gap:11px; align-items:flex-start; animation:rise .5s var(--eo) both;}
.warn span{color:var(--warn) !important;}
.warn svg{flex-shrink:0; margin-top:2px;}

/* alternatives */
.alt{display:flex; align-items:center; gap:14px; margin:12px 0; animation:rise .5s var(--eo) both;}
.alt .rank{width:26px; height:26px; flex-shrink:0; border-radius:8px; background:var(--cream2); color:var(--green) !important; font-weight:700; font-size:.82rem; display:grid; place-items:center;}
.alt .body{flex:1;}
.alt .top{display:flex; justify-content:space-between; font-size:.9rem; margin-bottom:6px;}
.alt .nm{font-weight:600;}
.alt .pc{color:var(--muted) !important; font-weight:600;}
.track{height:12px; background:var(--cream2); border-radius:6px; overflow:hidden;}
.track .fill{height:100%; border-radius:6px; width:var(--w); animation:grow 1s var(--eo);}
.f1{background:var(--green)} .f2{background:var(--green2)} .f3{background:var(--sage)}
@keyframes grow{from{width:0}to{width:var(--w)}}

/* XAI */
.xai-grid{display:grid; grid-template-columns:1fr 1fr; gap:32px;}
@media(max-width:600px){.xai-grid{grid-template-columns:1fr; gap:22px}}
.xai h4{font-size:.86rem; font-weight:700; margin:0 0 14px; color:var(--green) !important;}
.xrow{display:flex; align-items:center; gap:11px; margin:10px 0;}
.xrow .xn{width:118px; font-size:.82rem;}
.xrow .xb{flex:1; height:14px; background:var(--cream2); border-radius:5px; overflow:hidden;}
.xrow .xf{height:100%; border-radius:5px; width:var(--w); animation:grow 1s var(--eo);}
.xa{background:var(--green)} .xt{background:var(--green2)}
.xrow .xp{width:38px; text-align:right; font-size:.8rem; font-weight:600; color:var(--green2) !important;}

.disclaimer{margin-top:26px; padding-top:16px; border-top:1px solid var(--line); font-size:.82rem; color:var(--muted2) !important; line-height:1.6;}
</style>
""", unsafe_allow_html=True)

# =========================================================
#  KONFIG  (identik dengan versi kamu)
# =========================================================
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

# =========================================================
#  HERO
# =========================================================
st.markdown("""
<div class="hero">
  <div class="glow"></div>
  <div class="topo"></div>
  <div class="rings"><s></s><s></s><s></s><s></s><b></b></div>
  <div class="eyebrow"><span class="bar"></span>Presisi pertanian berbasis sensor</div>
  <h1>Keputusan tepat untuk<br>setiap <span class="soft">petak lahan.</span></h1>
  <p class="lead">Data tanah dan cuaca dibaca model TabNet & RF, lalu menjadi rekomendasi pupuk dan irigasi yang bisa dijelaskan.</p>
  
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="kick"><span class="bar"></span>Data sensor</div>', unsafe_allow_html=True)
st.markdown('<p class="sub">Masukkan pembacaan lahan, lalu jalankan model.</p>', unsafe_allow_html=True)

# =========================================================
#  INPUTS
# =========================================================
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
go = st.button("Dapatkan Rekomendasi")

# =========================================================
#  RESULT RENDERERS
# =========================================================
WARN_SVG = ('<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#6f5622" '
            'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M12 9v4M12 17h.01M10.3 3.9 2 18a2 2 0 0 0 1.7 3h16.6a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>')

def html(s): st.markdown(s, unsafe_allow_html=True)

def render_results(vals, fert, conf, mm, top3, xai_f, xai_i, narasi, dosis_txt, warns):
    liter = mm * LUAS_M2
    water_pct = min(mm / 12.0 * 100.0, 100.0)

    # header + chips
    html(f'''
    <div class="res-head">
      <div class="res-title">Hasil analisis lahan</div>
      <div class="chips">
        <span class="chip">Tanaman {vals.get("crop","")}</span>
        <span class="chip">Tanah {vals.get("soil_type","")}</span>
      </div>
    </div>''')

    # metric cards
    html(f'''
    <div class="metrics">
      <div class="metric">
        <div class="mc">
          <div class="k">Rekomendasi pupuk</div>
          <div class="v">{nice_fert(fert)}</div>
          <div class="s">Tingkat keyakinan model</div>
        </div>
        <div class="donut" style="--p:{conf*100:.0f}"><span class="dl">{conf*100:.0f}%</span></div>
      </div>
      <div class="metric">
        <div class="mc">
          <div class="k">Kebutuhan irigasi</div>
          <div class="v">{mm:.2f} <span class="unit">mm/hari</span></div>
          <div class="s">&plusmn; {liter:.0f} liter untuk 0,25 ha</div>
        </div>
        <div class="wlevel"><div class="fillw" style="--w:{water_pct:.0f}%"></div></div>
      </div>
    </div>''')

    # advice
    dose_html = f'<span class="dose">{dosis_txt}</span>' if dosis_txt else ''
    html(f'<div class="blk">Rekomendasi tindakan</div>'
         f'<div class="advice"><p>{narasi}</p>{dose_html}</div>')

    # warnings
    if warns:
        html('<div class="blk">Peringatan kondisi lahan</div>')
        for i, w in enumerate(warns):
            html(f'<div class="warn" style="animation-delay:{i*0.05:.2f}s">{WARN_SVG}<span>{w}</span></div>')

    # alternatives
    html('<div class="blk">Alternatif pupuk</div>')
    for idx, (name, p) in enumerate(top3, 1):
        html(f'<div class="alt" style="animation-delay:{(idx-1)*0.06:.2f}s">'
             f'<div class="rank">{idx}</div>'
             f'<div class="body"><div class="top"><span class="nm">{nice_fert(name)}</span>'
             f'<span class="pc">{p*100:.0f}%</span></div>'
             f'<div class="track"><div class="fill f{idx}" style="--w:{p*100:.0f}%"></div></div></div></div>')

    # XAI
    def xrows(pairs, fill_cls):
        out = ""
        for name, frac in pairs:
            out += (f'<div class="xrow"><div class="xn">{FEAT_ID.get(name,name)}</div>'
                    f'<div class="xb"><div class="xf {fill_cls}" style="--w:{frac*100:.0f}%"></div></div>'
                    f'<div class="xp">{frac*100:.0f}%</div></div>')
        return out
    html('<div class="blk">Penjelasan (Explainable AI)</div>')
    html(f'''<div class="xai-grid">
      <div class="xai"><h4>Faktor pupuk</h4>{xrows(xai_f,"xa")}</div>
      <div class="xai"><h4>Faktor irigasi</h4>{xrows(xai_i,"xt")}</div>
    </div>''')

    html('<p class="disclaimer">Dosis bersifat perkiraan dan wajib disesuaikan dengan '
         'rekomendasi penyuluh setempat. Model dilatih pada data sintetis untuk validasi '
         'pipeline; angka bukan hasil pengukuran lapangan.</p>')

# =========================================================
#  RUN
# =========================================================
if go:
    res = recommend(vals)
    st.markdown('<hr>', unsafe_allow_html=True)
    render_results(vals, *res)