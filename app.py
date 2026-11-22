"""
Demo: riconoscimento di banconote in euro con risposta vocale.
Avvio in locale:  streamlit run app.py
"""
import io
from collections import Counter

import streamlit as st
from PIL import Image, ImageDraw
from gtts import gTTS
import base64
import requests

# ---------------- CONFIGURAZIONE ----------------
MODEL_ID = "money-detection-2/money-detection-v3-1-yolo26s-t1"   # ID del progetto Roboflow / numero di versione
SOGLIA = 0.60                  # soglia di confidenza scelta sul validation set
USA_UPLOADER = False           # True se st.camera_input apre la fotocamera frontale

# Nomi delle classi nel dataset -> valore in CENTESIMI
VALORI = {"eur_5": 500, "eur_10": 1000, "eur_20": 2000,
          "eur_50": 5000, "eur_100": 10000, "eur_200": 20000,
          "eur_2": 200, "eur_1": 100,
          "cent_50": 50, "cent_20": 20, "cent_10": 10,
          "cent_5": 5, "cent_2": 2, "cent_1": 1}


# ---------------- FUNZIONI ----------------
def rileva(img):
    """Manda l'immagine al modello Roboflow e tiene solo le previsioni sopra soglia."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    risposta = requests.post(
        f"https://serverless.roboflow.com/{MODEL_ID}",
        params={"api_key": st.secrets["ROBOFLOW_API_KEY"]},
        data=base64.b64encode(buf.getvalue()).decode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if not risposta.ok:
        st.error(f"Errore Roboflow {risposta.status_code}: {risposta.text[:300]}")
        st.stop()
    tutte = risposta.json().get("predictions", [])
    sopra = [p for p in tutte if p["confidence"] >= SOGLIA]
    return tutte, sopra


def in_parole(c):
    """Importo in centesimi -> testo: 250 -> '2 euro e 50 centesimi'."""
    euro, cent = divmod(c, 100)
    cent_txt = "1 centesimo" if cent == 1 else f"{cent} centesimi"
    if euro and cent:
        return f"{euro} euro e {cent_txt}"
    return f"{euro} euro" if euro else cent_txt


def componi_messaggio(tutte, sopra):
    """Trasforma le previsioni in una frase breve, pensata per l'ascolto."""
    if not tutte:
        return "Nessun contante. Avvicina la fotocamera."
    valori = sorted((VALORI[p["class"]] for p in sopra if p["class"] in VALORI),
                    reverse=True)
    if not valori:
        return "Non sono sicuro. Riprova."
    if len(valori) == 1:
        return in_parole(valori[0])
    conteggio = Counter(valori)
    parti = [f"{n} da {in_parole(v)}" if n > 1 else in_parole(v)
             for v, n in sorted(conteggio.items(), reverse=True)]
    return f"{len(valori)} pezzi: {', '.join(parti)}. Totale {in_parole(sum(valori))}."


def parla(testo):
    """Converte il testo in audio mp3 con gTTS e lo riproduce nel browser."""
    buf = io.BytesIO()
    gTTS(testo, lang="it").write_to_fp(buf)
    st.audio(buf.getvalue(), format="audio/mp3", autoplay=True)


def disegna_box(img, previsioni):
    """Disegna le box sull'immagine (utile per la presentazione, non per l'utente)."""
    img = img.copy()
    d = ImageDraw.Draw(img)
    for p in previsioni:
        x0, y0 = p["x"] - p["width"] / 2, p["y"] - p["height"] / 2
        x1, y1 = p["x"] + p["width"] / 2, p["y"] + p["height"] / 2
        d.rectangle([x0, y0, x1, y1], outline="lime", width=5)
        d.text((x0 + 6, y0 + 6), f'{p["class"]} {p["confidence"]:.2f}', fill="lime")
    return img


# ---------------- INTERFACCIA ----------------
st.set_page_config(page_title="Che banconota è?", layout="centered")
st.markdown("""
<style>
  /* pulsante della fotocamera grande e facile da trovare */
  [data-testid="stCameraInput"] button { min-height: 6rem; font-size: 1.6rem; }
  .risultato { font-size: 3rem; font-weight: 700; text-align: center;
               padding: 1rem; border-radius: 12px; background: #000; color: #FFD400; }
</style>""", unsafe_allow_html=True)

st.title("Che banconota è?")

if USA_UPLOADER:
    foto = st.file_uploader("Scatta una foto della banconota", type=["jpg", "jpeg", "png"])
else:
    foto = st.camera_input("Inquadra la banconota e scatta")

if foto is not None:
    img = Image.open(foto).convert("RGB")
    img.thumbnail((1280, 1280))
    with st.spinner("Riconoscimento in corso..."):
        tutte, sopra = rileva(img)
    messaggio = componi_messaggio(tutte, sopra)

    # risultato: grande, ad alto contrasto, e letto ad alta voce
    st.markdown(f'<div class="risultato" aria-live="assertive">{messaggio}</div>',
                unsafe_allow_html=True)
    parla(messaggio)

    with st.expander("Dettagli per la presentazione"):
        st.image(disegna_box(img, tutte), use_container_width=True)
        st.json(tutte)
