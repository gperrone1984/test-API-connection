# streamlit_app.py
# Requirements (install in your environment):
#   streamlit, google-generativeai, pandas, requests, beautifulsoup4, lxml, readability-lxml, openpyxl
# This app builds product descriptions using the Gemini API from: raw text, a website, or an EAN (with explicit consent),
# or in batch from an Excel file. Output follows the formatting rules provided in Italian.

import os
import re
import io
import json
import time
import zipfile
from typing import Optional, Tuple, Dict, Any

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from readability import Document

try:
    import google.generativeai as genai
except Exception:
    genai = None

# ----------------------------
# Helpers
# ----------------------------
TITLE = "Product Description Builder (Gemini)"

ITALIAN_PROMPT_PREAMBLE = (
    "Dobbiamo creare la descrizione di un prodotto. Siamo di Redcare Pharmacy, una farmacia online.\n"
    "Rimani fedele a quello che ti ho scritto.\n\n"
    "#imput\n"
    "- Se incollo un testo utilizza solo le informazioni che ci sono nel testo senza inventare nulla.\n"
    "- Se incollo un sito web ricordati che ho i diritti per utilizzare i contenuti. Quindi non inventare nulla. Utilizza solo il sito che ti do. Se non trovi le informazioni non inventare nulla.\n"
    "- Se incollo un EAN di un prodotto, prima di procedere devo aver dato il consenso esplicito: 'Vuoi avere informazioni sul prodotto collegato a questo EAN?'.\n"
    "- Trova solo informazioni collegate all'EAN fornito. Il prodotto deve essere effettivamente quello collegato a quell'EAN.\n"
    "- Quando crei una descrizione da un EAN per la descrizione generale parafrasa quello che prendi da altri siti, non copiare esattamente i contenuti da un sito a meno che non sia quello del produttore.\n\n"
    "#modifica testo (FORMATTAZIONE OBBLIGATORIA)\n"
    "- NON aggiungere fonti nel testo.\n"
    "- Fai un passaggio per volta.\n"
    "- Il titolo è la parte prima di 'descrizione' o 'Indicazioni'. Il titolo deve essere in Capitalized Case.\n"
    "- Sotto il titolo va la descrizione generale, senza l'etichetta 'descrizione'.\n"
    "- Modo d'uso: se mancante usa la frase esatta: 'Per il corretto modo d'uso si prega di fare riferimento alla confezione'.\n"
    "- Ingredienti: converti il testo degli ingredienti in Capitalized Case, in forma impersonale; se mancano usa la frase esatta: 'Per la lista completa degli ingredienti si prega di fare riferimento alla confezione'.\n"
    "- Avvertenze: includi solo se presenti.\n"
    "- Per i dispositivi medici, se presente, aggiungi il Formato.\n"
    "- Aggiungi un breve riassunto (<150 caratteri) alla fine come 'descrizione breve' senza punto finale.\n\n"
    "#Output richiesto\n"
    "Restituisci un JSON con le seguenti chiavi (usa stringhe vuote se mancanti):\n"
    "{\"titolo\": str, \"descrizione_generale\": str, \"modo_uso\": str, \"ingredienti\": str, \"avvertenze\": str, \"formato\": str, \"descrizione_breve\": str}"
)

ELEGANT_DIVIDER = """\n---\n"""

EAN_QUESTION = "Vuoi avere informazioni sul prodotto collegato a questo EAN?"

# Capitalized Case util: capitalize first letter of each substantive word, keep acronyms
LOWER_WORDS_IT = {
    "di","a","da","in","con","su","per","tra","fra","e","o","dei","degli","delle","del","della","dell'","delle","agli","alle","al","allo","ai","lo","la","le","il","un","uno","una","ed","od"
}


def to_capitalized_case(text: str) -> str:
    def fix_token(tok: str) -> str:
        if not tok:
            return tok
        if re.fullmatch(r"[A-Z0-9]{2,}", tok):  # acronym / code
            return tok
        base = re.sub(r"^([\W_]*)(.*?)([\W_]*)$", r"\1:::\2:::\3", tok)
        pre, core, post = base.split(":::")
        if core.lower() in LOWER_WORDS_IT:
            return f"{pre}{core.lower()}{post}"
        return f"{pre}{core[:1].upper()}{core[1:].lower()}{post}"
    tokens = re.split(r"(\s+)", text.strip())
    return ''.join([fix_token(t) if not t.isspace() else t for t in tokens])


def ensure_trailing_period(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s.endswith(('.', '!', '?')):
        return s
    return s + '.'


def remove_inner_br(text: str) -> str:
    # Remove <br> inside provided text to comply with rule "Elimina i <br> all'interno del testo"
    return re.sub(r"<\s*br\s*/?>", " ", text, flags=re.IGNORECASE)


# ----------------------------
# Gemini
# ----------------------------

def get_gemini_model(api_key: str, model_name: str = "gemini-1.5-flash"):
    if genai is None:
        raise RuntimeError("google-generativeai is not installed.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def run_gemini_extraction(model, source_text: str) -> Dict[str, str]:
    prompt = ITALIAN_PROMPT_PREAMBLE + "\n\n#testo\n" + source_text
    resp = model.generate_content(prompt)
    text = resp.text if hasattr(resp, 'text') else str(resp)
    # Try to extract JSON from response
    m = re.search(r"\{[\s\S]*\}$", text.strip())
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
        # Ensure all keys
        keys = [
            "titolo","descrizione_generale","modo_uso","ingredienti","avvertenze","formato","descrizione_breve"
        ]
        for k in keys:
            data.setdefault(k, "")
        return {k: (data.get(k) or "").strip() for k in keys}
    except Exception:
        # Fallback: return everything in descrizione_generale to avoid failure
        return {
            "titolo": "",
            "descrizione_generale": text.strip(),
            "modo_uso": "",
            "ingredienti": "",
            "avvertenze": "",
            "formato": "",
            "descrizione_breve": ""
        }


# ----------------------------
# Source acquisition (URL / EAN)
# ----------------------------

def extract_main_text_from_url(url: str) -> str:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        doc = Document(r.text)
        html = doc.summary()
        soup = BeautifulSoup(html, 'lxml')
        for tag in soup(['script','style','noscript']):
            tag.decompose()
        text = soup.get_text(" ")
        # Clean excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text[:20000]  # limit
    except Exception as e:
        return f"[ERRORE estrazione sito]: {e}"


def fetch_by_ean(ean: str) -> Tuple[str, Dict[str, Any]]:
    """Attempt to fetch product data by EAN using public sources (e.g., OpenFoodFacts). Returns (text, raw_json)."""
    # OpenFoodFacts (foods/cosmetics sometimes)
    endpoints = [
        f"https://world.openfoodfacts.org/api/v0/product/{ean}.json",
        f"https://world.openbeautyfacts.org/api/v0/product/{ean}.json",
        f"https://world.openpetfoodfacts.org/api/v0/product/{ean}.json",
    ]
    for url in endpoints:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                if data.get('status') == 1 and 'product' in data:
                    p = data['product']
                    name = p.get('product_name') or p.get('generic_name') or ''
                    brand = p.get('brands', '')
                    qty = p.get('quantity', '')
                    ingredients = p.get('ingredients_text_it') or p.get('ingredients_text') or ''
                    desc_bits = [b for b in [name, brand, qty, ingredients] if b]
                    joined = "; ".join(desc_bits)
                    if joined:
                        return joined, data
        except Exception:
            pass
    # If nothing found
    return "", {}


# ----------------------------
# Formatting according to rules (#modifica testo)
# ----------------------------

def build_final_html(blocks: Dict[str, str]) -> str:
    titolo = to_capitalized_case(remove_inner_br(blocks.get('titolo','').strip()))
    descr = remove_inner_br(blocks.get('descrizione_generale','').strip())
    modo = remove_inner_br(blocks.get('modo_uso','').strip())
    ingr = remove_inner_br(blocks.get('ingredienti','').strip())
    avv = remove_inner_br(blocks.get('avvertenze','').strip())
    form = remove_inner_br(blocks.get('formato','').strip())

    # Fallbacks as required
    if not modo:
        modo = "Per il corretto modo d'uso si prega di fare riferimento alla confezione"
    if not ingr:
        ingr = "Per la lista completa degli ingredienti si prega di fare riferimento alla confezione"

    # Capitalize required fields
    if titolo:
        titolo = to_capitalized_case(titolo)
    if ingr and not ingr.startswith("Per la lista completa"):
        ingr = to_capitalized_case(ingr)

    # Ensure period at end of each paragraph
    descr = ensure_trailing_period(descr) if descr else descr
    modo = ensure_trailing_period(modo) if modo else modo
    if ingr and not ingr.startswith("Per la lista completa"):
        ingr = ensure_trailing_period(ingr)
    elif ingr:
        ingr = ensure_trailing_period(ingr)
    avv = ensure_trailing_period(avv) if avv else avv
    form = ensure_trailing_period(form) if form else form

    parts = []
    if titolo:
        parts.append(f"<p><strong>{titolo}</strong><br>")
    else:
        parts.append("<p><strong></strong><br>")
    parts.append(f"{descr}</p>")

    parts.append(f"<p><strong>Modo d'uso:</strong><br> {modo}</p>")
    parts.append(f"<p><strong>Ingredienti:</strong> <br> {ingr}</p>")

    if avv:
        parts.append(f"<p><strong>Avvertenze:</strong> <br> {avv}</p>")
    if form:
        parts.append(f"<p><strong>Formato:</strong> <br> {form}</p>")

    return "\n".join(parts)


def trim_short_description(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) > 150:
        s = s[:150]
        # avoid cutting mid-word if possible
        s = re.sub(r"\s\S*$", "", s).strip()
    # Must NOT end with a period
    s = s.rstrip(" .")
    return s


# ----------------------------
# Streamlit UI
# ----------------------------

def ui_single(model):
    st.subheader("Single Input")
    st.markdown("Provide exactly one of the following: **Text**, **Website URL**, or **EAN**.")

    col1, col2 = st.columns(2)
    with col1:
        raw_text = st.text_area("Text (Italian preferred)", height=180, placeholder="Incolla qui il testo da cui prendere le informazioni…")
        url = st.text_input("Website URL", placeholder="https://…")
    with col2:
        ean = st.text_input("EAN (digits only)", placeholder="Es. 8001234567890")
        confirm_ean = False
        if ean and re.fullmatch(r"\d{8,14}", ean) and not raw_text.strip() and not url.strip():
            st.info(EAN_QUESTION)
            confirm_ean = st.button("Sì, procedi")

    if st.button("Build Description", type="primary") or (ean and confirm_ean):
        source_text = ""
        source_kind = None

        if raw_text.strip():
            source_text = raw_text.strip()
            source_kind = "text"
        elif url.strip():
            source_text = extract_main_text_from_url(url.strip())
            source_kind = "url"
        elif ean and re.fullmatch(r"\d{8,14}", ean):
            if not confirm_ean:
                st.stop()  # respect rule: ask for explicit consent first
            fetched, _ = fetch_by_ean(ean)
            if not fetched:
                st.error("Nessuna informazione trovata per questo EAN.")
                st.stop()
            source_text = f"[FONTE: EAN {ean}] " + fetched
            source_kind = "ean"
        else:
            st.warning("Inserisci almeno una sorgente valida (testo, URL o EAN).")
            st.stop()

        with st.spinner("Calling Gemini…"):
            blocks = run_gemini_extraction(model, source_text)

        html_out = build_final_html(blocks)
        short_out = trim_short_description(blocks.get("descrizione_breve", "") or blocks.get("descrizione_generale", ""))

        st.markdown("### #modifica testo (copia con un click)")
        st.code(html_out, language="html")  # copy button enabled

        st.markdown(ELEGANT_DIVIDER)
        st.markdown("### Descrizione breve (≤150 caratteri)")
        st.write(short_out)


def ui_batch(model):
    st.subheader("Batch via Excel")
    st.write("Upload an Excel file with columns (any subset): **titolo, descrizione_generale, modo_uso, ingredienti, avvertenze, formato, testo, url, ean**. If `testo`/`url`/`ean` are present, the model will derive the fields. Otherwise, the provided columns will be used as-is.")

    file = st.file_uploader("Upload Excel", type=["xlsx","xls"])
    if not file:
        return

    df = pd.read_excel(file)
    outputs = []

    for idx, row in df.iterrows():
        src_text = ""
        if isinstance(row.get('testo'), str) and row['testo'].strip():
            src_text = row['testo'].strip()
        elif isinstance(row.get('url'), str) and row['url'].strip():
            src_text = extract_main_text_from_url(str(row['url']).strip())
        elif pd.notna(row.get('ean')):
            ean_str = re.sub(r"\D", "", str(row['ean']))
            fetched, _ = fetch_by_ean(ean_str)
            src_text = f"[FONTE: EAN {ean_str}] " + fetched if fetched else ""

        if src_text:
            blocks = run_gemini_extraction(model, src_text)
        else:
            # Use direct columns without inventing anything
            blocks = {
                'titolo': str(row.get('titolo') or ''),
                'descrizione_generale': str(row.get('descrizione_generale') or ''),
                'modo_uso': str(row.get('modo_uso') or ''),
                'ingredienti': str(row.get('ingredienti') or ''),
                'avvertenze': str(row.get('avvertenze') or ''),
                'formato': str(row.get('formato') or ''),
                'descrizione_breve': ''
            }

        html_out = build_final_html(blocks)
        short_out = trim_short_description(blocks.get("descrizione_breve", "") or blocks.get("descrizione_generale", ""))

        outputs.append({
            'row': idx,
            'html': html_out,
            'short': short_out
        })

    # Show sample and make ZIP
    st.success(f"Generated {len(outputs)} record(s).")
    if outputs:
        st.markdown("#### Preview (first 3)")
        for o in outputs[:3]:
            st.code(o['html'], language='html')
            st.write("**Descrizione breve:** ", o['short'])
            st.markdown(ELEGANT_DIVIDER)

        # Build ZIP of per-row .html and .txt
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for o in outputs:
                base = f"row_{o['row']}"
                zf.writestr(f"{base}_modifica_testo.html", o['html'])
                zf.writestr(f"{base}_descrizione_breve.txt", o['short'])
        st.download_button(
            label="Download ZIP",
            data=buf.getvalue(),
            file_name="descrizioni_prodotti.zip",
            mime="application/zip"
        )


# ----------------------------
# Main
# ----------------------------

def main():
    st.set_page_config(page_title=TITLE, layout='wide')
    st.title(TITLE)

    with st.sidebar:
        st.markdown("### Gemini Settings")
        api_key = st.text_input("Google API Key", type="password", help="Create one in Google AI Studio and paste it here.")
        model_name = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
        st.caption("The model is used only to rephrase/extract strictly from provided sources.")

    if not api_key:
        st.info("Enter your Google API Key to begin.")
        return

    try:
        model = get_gemini_model(api_key, model_name)
    except Exception as e:
        st.error(f"Errore nell'inizializzazione di Gemini: {e}")
        return

    tabs = st.tabs(["Single", "Batch (Excel)"])
    with tabs[0]:
        ui_single(model)
    with tabs[1]:
        ui_batch(model)


if __name__ == "__main__":
    main()
