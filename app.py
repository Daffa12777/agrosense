import json
import re
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from llm_narrator import narrate


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
.ai-note{margin-top:18px; background:var(--white); border:1px solid var(--line); border-left:4px solid var(--sage); border-radius:16px; padding:16px 18px; animation:rise .6s var(--eo) both;}
.ai-note p{margin:8px 0 0; font-size:.98rem; line-height:1.65; color:var(--text) !important;}
.ai-badge{display:inline-block; background:var(--green2); color:#fff !important; font-size:.72rem; font-weight:700; padding:4px 10px; border-radius:999px; letter-spacing:.02em;}

/* ---------- AI action steps ---------- */
.steps{list-style:none; margin:6px 0 0; padding:0;}
.steps li{position:relative; padding:13px 16px 13px 44px; margin:9px 0; background:var(--white);
  border:1px solid var(--line); border-radius:12px; font-size:.96rem; line-height:1.6;
  color:var(--text) !important; animation:rise .5s var(--eo) both;}
.steps li .n{position:absolute; left:12px; top:13px; width:22px; height:22px; border-radius:6px;
  background:var(--green); color:#fff !important; font-size:.72rem; font-weight:700; display:grid; place-items:center;}

/* ---------- skeleton loading (saran tindakan) ---------- */
.sk-wrap{margin:6px 0 0;}
.sk-load{display:flex; align-items:center; gap:10px; color:var(--muted) !important; font-size:.88rem; margin-bottom:12px;}
.sk-dot{width:15px; height:15px; border:2px solid var(--cream2); border-top-color:var(--green2); border-radius:50%; animation:spin .8s linear infinite;}
@keyframes spin{to{transform:rotate(360deg)}}
.sk-bar{height:48px; border-radius:12px; margin:9px 0; border:1px solid var(--line);
  background:linear-gradient(90deg,var(--cream2) 25%,#f2ecdd 37%,var(--cream2) 63%);
  background-size:400% 100%; animation:shine 1.3s ease infinite;}
@keyframes shine{0%{background-position:100% 0}100%{background-position:-100% 0}}
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

GEMINI_MODEL = "gemini-3.8-flash"   # ganti ke flash terbaru bila perlu

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
#  SARAN TINDAKAN DINAMIS  (Gemini API + fallback aman)
# =========================================================
def _fallback_steps(fert, mm, dosis_txt, warns, vals):
    """Saran rule-based. SELALU jalan, dipakai kalau LLM offline/error/halusinasi."""
    steps = []
    if fert == "None":
        steps.append("Belum perlu pemupukan saat ini; pertahankan kondisi hara tanah dan lanjutkan perawatan rutin.")
    else:
        s = f"Berikan {nice_fert(fert)} sesuai kebutuhan tanaman."
        if dosis_txt: s += f" {dosis_txt}."
        steps.append(s)
    if mm < 1:
        steps.append("Tunda penyiraman terlebih dahulu karena tanah masih cukup lembap; cek kembali kelembapan esok hari.")
    else:
        liter = mm * LUAS_M2
        steps.append(f"Siram sekitar {mm:.2f} mm (kurang lebih {liter:.0f} liter untuk lahan 0,25 ha), sebaiknya pada pagi hari agar penyerapan optimal.")
    for w in warns[:3]:
        steps.append(w)
    steps.append("Konsultasikan kembali ke penyuluh pertanian setempat sebelum aplikasi di lapangan.")
    return steps[:6]

def _allowed_numbers(mm, dosis_txt):
    """Angka yang boleh muncul di output LLM (anti-ngarang dosis/irigasi)."""
    nums = set(re.findall(r"\d+", dosis_txt or ""))
    nums |= {f"{mm:.2f}", f"{mm:.1f}", str(int(round(mm)))}
    nums.add(str(int(LUAS_M2)))
    nums.add(f"{mm*LUAS_M2:.0f}")
    return nums

def _valid_steps(text, allowed):
    """Tolak kalau ada angka satuan (kg/mm/liter/%) yang tidak ada di input."""
    for num in re.findall(r"(\d+(?:[.,]\d+)?)\s*(?:kg|mm|liter|l|%)", text.lower()):
        clean = num.replace(",", ".")
        if clean not in allowed and clean.rstrip("0").rstrip(".") not in {a.rstrip("0").rstrip(".") for a in allowed}:
            return False
    return True

@st.cache_data(show_spinner=False)
def _ask_gemini(prompt: str, allowed: tuple):
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        return None
    import time
    from google import genai
    client = genai.Client(api_key=api_key)
    last_err = None
    for attempt in range(4):
        try:
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            text = (getattr(resp, "text", "") or "").strip()
            if text and _valid_steps(text, set(allowed)):
                return text
            return None
        except Exception as e:
            last_err = e
            if "503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e):
                time.sleep(2 * (attempt + 1))   # 2s,4s,6s
                continue
            break
    st.warning(f"Gemini gagal: {last_err}")
    return None

def advise_actions(fert, conf, mm, dosis_txt, warns, vals, top3):
    """
    return (steps_list, sumber). sumber: 'ai' | 'otomatis'. Tidak pernah gagal.
    LLM hanya MERANGKAI saran; angka dosis/irigasi tetap dari model & rule.
    """
    fb = _fallback_steps(fert, mm, dosis_txt, warns, vals)

    def _fmt(k, v):
        if v is None: return None
        return f"{v:.1f}" if k == "soil_ph" else f"{v:.0f}"
    kondisi = ", ".join(f"{FEAT_ID.get(k,k)} {_fmt(k, vals.get(k))}"
                        for k in NUM if vals.get(k) is not None)

    prompt = f"""Kamu penyuluh pertanian profesional yang memberi arahan tindakan kepada petani.
Berdasarkan hasil analisis lahan di bawah, susun 4-6 langkah tindakan yang JELAS, DETAIL, dan PROFESIONAL dalam bahasa Indonesia.

GAYA:
- Tiap langkah 1-2 kalimat: sebutkan APA yang dilakukan, BERAPA (bila ada angkanya), KAPAN/CARA, dan ALASAN singkatnya.
- Bahasa sopan dan mudah dipahami petani, tetapi terdengar seperti anjuran penyuluh, bukan catatan singkat.
- Urutkan logis: pemupukan, irigasi, penanganan kondisi lahan/peringatan, lalu pemantauan.

ATURAN KERAS:
- Untuk angka dosis/irigasi/liter/persen, gunakan HANYA angka yang tertera di bawah. Dilarang membuat angka baru.
- Dilarang menyebut merek pupuk/pestisida atau klaim di luar data. Jangan sebut kata "model", "SHAP", atau istilah teknis.
- Keluarkan HANYA daftar langkah, satu langkah per baris, tanpa penomoran.

DATA ANALISIS LAHAN:
- Tanaman: {vals.get("crop","")}; Jenis tanah: {vals.get("soil_type","")}
- Rekomendasi pupuk: {nice_fert(fert)} (tingkat keyakinan {conf*100:.0f}%)
- Dosis pupuk anjuran: {dosis_txt or "tidak ada / belum diperlukan"}
- Kebutuhan irigasi: {mm:.2f} mm/hari (setara {mm*LUAS_M2:.0f} liter untuk lahan 0,25 ha)
- Kondisi lahan terukur: {kondisi}
- Peringatan kondisi: {" | ".join(warns) if warns else "tidak ada peringatan khusus"}
"""
    allowed = tuple(sorted(_allowed_numbers(mm, dosis_txt)))
    out = _ask_gemini(prompt, allowed)
    if not out:
        return fb, "otomatis"
    steps = []
    for line in out.splitlines():
        s = line.strip().lstrip("-•*0123456789. )").strip()
        if s:
            steps.append(s)
    steps = steps[:6]
    return (steps, "ai") if steps else (fb, "otomatis")

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

def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

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
    html(f'<div class="blk">Ringkasan keputusan</div>'
         f'<div class="advice"><p>{narasi}</p>{dose_html}</div>')

    # warnings
    if warns:
        html('<div class="blk">Langkah tindakan (detail)</div>')
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

        # ---- Narasi LLM (aman: template kalau offline/error/halusinasi) ----
    html('<div class="blk">Ringkasan untuk petani</div>')
    ph_narasi = st.empty()
    ph_narasi.markdown(
        '<div class="sk-wrap"><div class="sk-load"><span class="sk-dot"></span>'
        'Menyusun ringkasan...</div>'
        '<div class="sk-bar"></div><div class="sk-bar"></div></div>',
        unsafe_allow_html=True)
    ai_text, src = narrate(nice_fert(fert), conf, mm, xai_f, xai_i, FEAT_ID)
    badge = "Dijelaskan AI" if src == "ai" else "Ringkasan otomatis"
    ph_narasi.markdown(
        f'<div class="ai-note"><span class="ai-badge">{badge}</span><p>{ai_text}</p></div>',
        unsafe_allow_html=True)

    # ---- Saran tindakan dinamis (Gemini + fallback) dengan loading skeleton ----
    html('<div class="blk">Langkah Tindakan (detail)</div>')
    ph = st.empty()
    ph.markdown(
        '<div class="sk-wrap"><div class="sk-load"><span class="sk-dot"></span>'
        'Menyusun saran tindakan...</div>'
        '<div class="sk-bar"></div><div class="sk-bar"></div>'
        '<div class="sk-bar"></div><div class="sk-bar"></div></div>',
        unsafe_allow_html=True)
    steps, ssrc = advise_actions(fert, conf, mm, dosis_txt, warns, vals, top3)
    sbadge = "Disusun AI" if ssrc == "ai" else "Saran otomatis"
    items = "".join(f'<li style="animation-delay:{i*0.05:.2f}s"><span class="n">{i+1}</span>{esc(s)}</li>'
                    for i, s in enumerate(steps))
    ph.markdown(f'<span class="ai-badge">{sbadge}</span><ul class="steps">{items}</ul>',
                unsafe_allow_html=True)

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