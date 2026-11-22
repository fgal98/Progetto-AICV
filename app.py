"""
Demo: riconoscimento di banconote in euro con risposta vocale.
Avvio in locale:  streamlit run app.py
"""
import io
from collections import Counter

import streamlit as st
from PIL import Image, ImageDraw
from gtts import gTTS
from inference_sdk import InferenceHTTPClient

# ---------------- CONFIGURAZIONE ----------------
MODEL_ID = "nome-progetto/1"   # ID del progetto Roboflow / numero di versione
SOGLIA = 0.60                  # soglia di confidenza scelta sul validation set
USA_UPLOADER = False           # True se st.camera_input apre la fotocamera frontale

# Nomi delle classi nel dataset -> valore in euro
VALORI = {"eur_5": 5, "eur_10": 10, "eur_20": 20,
          "eur_50": 50, "eur_100": 100, "eur_200": 200}

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=st.secrets["ROBOFLOW_API_KEY"],
)


# ---------------- FUNZIONI ----------------
def rileva(img):
    """Chiama il modello e tiene solo le previsioni sopra soglia."""
    risultato = client.infer(img, model_id=MODEL_ID)
    tutte = risultato.get("predictions", [])
    sopra = [p for p in tutte if p["confidence"] >= SOGLIA]
    return tutte, sopra


def componi_messaggio(tutte, sopra):
    """Trasforma le previsioni in una frase breve, pensata per l'ascolto."""
    if not tutte:
        return "Nessuna banconota. Avvicina la fotocamera."
    if not sopra:
        return "Non sono sicuro. Riprova."
    valori = sorted(VALORI[p["class"]] for p in sopra if p["class"] in VALORI)
    if len(valori) == 1:
        return f"{valori[0]} euro"
    conteggio = Counter(valori)
    parti = [f"{n} da {v}" if n > 1 else f"{v}" for v, n in sorted(conteggio.items())]
    return f"{len(valori)} banconote: {', '.join(parti)} euro. Totale {sum(valori)} euro."


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
