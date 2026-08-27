# AgroSense LoRa-X — Demo

Rekomendasi pemupukan (klasifikasi) dan irigasi (regresi) dari data sensor tanah,
model TabNet. Antarmuka Streamlit.

## Jalankan lokal
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy Streamlit Cloud
1. Push repo ini ke GitHub (pastikan folder `models/` ikut, `*.zip` tidak di-ignore).
2. Buka share.streamlit.io -> New app -> pilih repo -> main file `app.py` -> Deploy.

## Struktur
```
app.py              UI + inferensi
train.py            latih ulang + regenerate models/
requirements.txt
models/
  pre_fert.joblib   preprocessor pupuk
  pre_irr.joblib    preprocessor irigasi
  label_encoder.joblib
  tabnet_fert.zip   model klasifikasi pupuk
  tabnet_irr.zip    model regresi irigasi
  meta.json         fitur, kelas, opsi form
```

Model dilatih pada data sintetis untuk validasi pipeline; ganti dengan data
lapangan sebelum penggunaan nyata.
