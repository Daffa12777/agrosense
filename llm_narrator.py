# llm_narrator.py — narasi aman berlapis di atas SHAP (template -> LLM -> validasi -> fallback)
import re
import streamlit as st

GEMINI_MODEL = "gemini-flash-latest"   # ganti ke flash terbaru bila perlu; kalau salah -> otomatis fallback template

def _pct(frac):            # 0..1 -> int persen
    return int(round(float(frac) * 100))

def _named(pairs, feat_id, k=3):
    return [(feat_id.get(key, key), _pct(frac)) for key, frac in list(pairs)[:k]]

# ---------- LAPIS 1: template deterministik (SELALU benar, offline-safe) ----------
def build_template(fert_label, conf, mm, xai_f, xai_i, feat_id):
    ff = _named(xai_f, feat_id, 3)
    fi = _named(xai_i, feat_id, 3)
    f_txt = ", ".join(f"{n} ({p}%)" for n, p in ff)
    i_txt = ", ".join(f"{n} ({p}%)" for n, p in fi)
    return (f'Rekomendasi "{fert_label}" (keyakinan {_pct(conf)}%) paling dipengaruhi oleh {f_txt}. '
            f'Kebutuhan irigasi {mm:.2f} mm/hari terutama dipengaruhi oleh {i_txt}.')

# ---------- LAPIS 3: validasi (anti-halusinasi) ----------
def _allowed(conf, mm, xai_f, xai_i):
    nums = {str(_pct(conf)), f"{mm:.2f}", f"{mm:.1f}", str(int(round(mm)))}
    for _, frac in list(xai_f)[:3] + list(xai_i)[:3]:
        nums.add(str(_pct(frac)))
    return nums

def _valid(text, allowed):
    # tolak kalau ada angka persen yg TIDAK ada di input
    for m in re.findall(r"(\d+)\s*%", text):
        if m not in allowed:
            return False
    return True

# ---------- LAPIS 2: LLM (opsional, cuma memperhalus) ----------
@st.cache_data(show_spinner=False)
def _ask_llm(prompt: str, allowed: tuple):
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        resp = genai.GenerativeModel(MODEL).generate_content(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        if text and _valid(text, set(allowed)):
            return text
        return None
    except Exception:
        return None

def narrate(fert_label, conf, mm, xai_f, xai_i, feat_id):
    """return (teks, sumber). sumber: 'ai' | 'template'. Tidak pernah gagal."""
    template = build_template(fert_label, conf, mm, xai_f, xai_i, feat_id)
    ff = _named(xai_f, feat_id, 3); fi = _named(xai_i, feat_id, 3)
    prompt = f"""Kamu penerjemah hasil model pertanian untuk petani awam.
Tulis ulang data berikut menjadi 2-3 kalimat bahasa Indonesia yang ramah, jelas, dan mudah dimengerti petani.

ATURAN KERAS:
- Sebutkan 2 faktor terpenting untuk pupuk dan 2 untuk irigasi.
- Jika menuliskan angka persen atau mm, gunakan HANYA angka yang tertera di bawah. Dilarang membuat angka baru.
- Dilarang menambah saran/dosis/klaim di luar data. Jangan sebut kata "SHAP" atau "model".
- Maksimal 3 kalimat.

Rekomendasi pupuk: {fert_label} (keyakinan {_pct(conf)}%)
Faktor pupuk: {ff}
Kebutuhan irigasi: {mm:.2f} mm/hari
Faktor irigasi: {fi}"""
    allowed = tuple(sorted(_allowed(conf, mm, xai_f, xai_i)))
    ai = _ask_llm(prompt, allowed)
    return (ai, "ai") if ai else (template, "template")
