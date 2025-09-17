import os
import re
import io
import json
import zipfile
from typing import Optional, Tuple, Dict, Any, List

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from readability import Document
from textwrap import dedent

try:
    import google.generativeai as genai
except Exception:
    genai = None

TITLE = "Product Description Builder (Gemini)"
ELEGANT_DIVIDER = "\n---\n"
EAN_QUESTION = "Vuoi avere informazioni sul prodotto collegato a questo EAN?"

LOWER_WORDS_IT = {
    "di","a","da","in","con","su","per","tra","fra","e","o","dei","degli","delle",
    "del","della","dell'","agli","alle","al","allo","ai","lo","la","le","il","un",
    "uno","una","ed","od"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36"
}

# Prompt completo
ITALIAN_PROMPT_PREAMBLE = dedent("""\
Dobbiamo creare la descrizione di un prodotto. Siamo di Redcare Pharmacy, una farmacia online.
- Rimani fedele a quello che ti ho scritto.

#input
- Incollare in chat il testo da cui prendere le informazioni.
- Se incollo un testo utilizza solo le informazioni che ci sono nel testo senza inventare nulla.
- Se incollo un sito web ricordati che ho i diritti per utilizzare i contenuti. Quindi non inventare nulla. Utilizza solo il sito che ti dò. Se non trovi le informazioni non inventare nulla.
- Se incollo un EAN di un prodotto NON SCRIVERE NULLA: Prima chiedimi se voglio cercare le informazioni del prototto collegate all`EAN. La domanda da fare è ESCLUSIVAMENTE: "Vuoi avere informazioni sul prodotto collegato a questo EAN?"
- Trova tutte le informazioni che riesci online collegate SOLO all´EAN che ti fornisco. Il prodotto deve essere effettivamente quello collegato all´EAN.
- Quando crei una descrizione da un EAN per la descrizione generale parafrasa quello che prendi da altri siti, non posso copiare esattamente i contenuti da un sito a meno che non sia quello del produttore.

#modifica testo
<p><strong> titolo </strong><br> dopo del titolo. Elimina i <br> all'interno del testo. Il testo deve essere in Capitalized Case. 
-Sotto il titolo va la descrizione generale del prodotto. Direttamente sotto il titolo, senza scrivere descrizione. Alla fine aggiungi </p>
-In Modo d'uso: devi prendere il testo di Modalità d'uso. Se la modalità d'uso/modo dúso manca utilizza la seguente frase: "Per il corretto modo d'uso si prega di fare riferimento alla confezione". Aggiungi <p><strong> prima di Modo d'uso: e </strong><br> alla fine. Alla fine di tutto metti </p>
-In Ingredienti: vanno Ingredienti o componenti. Converti il testo degli ingredienti in capitalized case.  Gli ingredienti devono essere in una forma impersonale. Se gli ingredienti mancano inserire la frase: Per la lista completa degli ingredienti si prega di fare riferimento alla confezione.  Aggiungi <p><strong> prima di Ingredienti: e </strong> <br> alla fine. Alla fine di tutto metti </p>. Converti il testo degli ingredienti in capitalized case. Non mettere in capitalized case la frase: Per la lista completa degli ingredienti si prega di fare riferimento alla confezione.
In Avvertenze: vanno Avvertenze. Aggiungi <p><strong> prima di Avvertenze: e </strong> <br> alla fine. Alla fine di tutto metti </p>. Se le Avvertenze non ci sono non scrivere nulla.
Per i dispositivi medici se presenti aggiungi il Formato, Aggiungi <p><strong> prima di Formato: e </strong> <br> alla fine. Alla fine di tutto metti </p>. Se il Formato non c´é non scrivere nulla.

#ISTRUZIONI DI OUTPUT
- Restituisci SOLO i tag HTML finali come specificato sopra (nessun JSON, nessun testo extra, nessun commento).
""")

# ----------------------------
# Utils
# ----------------------------
def to_capitalized_case(text: str) -> str:
    def fix_token(tok: str) -> str:
        if not tok:
            return tok
        if re.fullmatch(r"[A-Z0-9]{2,}", tok):
            return tok
        base = re.sub(r"^([\\W_]*)(.*?)([\\W_]*)$", r"\\1:::\\2:::\\3", tok)
        pre, core, post = base.split(":::")
        if core.lower() in LOWER_WORDS_IT:
            return f"{pre}{core.lower()}{post}"
        return f"{pre}{core[:1].upper()}{core[1:].lower()}{post}"
    tokens = re.split(r"(\\s+)", text.strip())
    return ''.join([fix_token(t) if not t.isspace() else t for t in tokens])

def get_api_key_from_env_or_secrets() -> Optional[str]:
    if 'GOOGLE_API_KEY' in st.secrets:
        return st.secrets['GOOGLE_API_KEY']
    return os.getenv('GOOGLE_API_KEY')

def get_gemini_model(api_key: str, model_name: str = "gemini-1.5-flash"):
    if genai is None:
        raise RuntimeError("google-generativeai non è installato.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

# ----------------------------
# Gemini
# ----------------------------
def run_gemini_to_html(model, source_text: str) -> str:
    prompt = f"{ITALIAN_PROMPT_PREAMBLE}\n\n{source_text}"
    resp = model.generate_content(prompt, request_options={"timeout": 60})
    text = getattr(resp, 'text', None) or str(resp)
    parts = re.findall(r"<p[\\s\\S]*?</p>", text, flags=re.IGNORECASE)
    html = " ".join(parts).strip() if parts else text.strip()
    return html

# ----------------------------
# Web Search by EAN
# ----------------------------
def duckduckgo_search_urls(query: str, max_results: int = 6) -> List[str]:
    try:
        resp = requests.post("https://duckduckgo.com/html/",
                             data={"q": query}, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        s = BeautifulSoup(resp.text, 'lxml')
        urls: List[str] = []
        for a in s.select("a.result__a"):
            href = a.get('href', '')
            if href and href.startswith("http"):
                urls.append(href)
            if len(urls) >= max_results:
                break
        return urls
    except Exception:
        return []

def extract_main_text_from_url(url: str) -> str:
    try:
        r = requests.get(url, timeout=25, headers=HEADERS)
        r.raise_for_status()
        doc = Document(r.text)
        html = doc.summary()
        soup = BeautifulSoup(html, 'lxml')
        for tag in soup(['script','style','noscript']):
            tag.decompose()
        text = soup.get_text(" ")
        return re.sub(r"\\s+", " ", text).strip()
    except Exception as e:
        return f"[ERRORE estrazione sito]: {e}"

def fetch_by_ean(ean: str) -> Tuple[str, Dict[str, str]]:
    queries = [f'"{ean}" sito ufficiale', str(ean)]
    urls = []
    for q in queries:
        res = duckduckgo_search_urls(q)
        for u in res:
            if u not in urls:
                urls.append(u)
        if len(urls) >= 8:
            break
    texts = []
    for u in urls[:8]:
        try:
            t = extract_main_text_from_url(u)
            if len(t) > 300:
                texts.append(t)
        except Exception:
            continue
    return ("\n\n".join(texts), {"source": "web"})

# ----------------------------
# Short description from HTML
# ----------------------------
def extract_descr_from_html(html: str) -> str:
    try:
        m = re.search(r"<p>\\s*<strong>.*?</strong><br>\\s*(.*?)</p>",
                      html, flags=re.IGNORECASE|re.DOTALL)
        if not m:
            return ""
        frag = m.group(1)
        txt = BeautifulSoup(frag, "lxml").get_text(" ")
        return re.sub(r"\\s+", " ", txt).strip()
    except Exception:
        return ""

def make_short_description(descr: str) -> str:
    s = re.sub(r"\\s+", " ", (descr or "").strip())
    if len(s) <= 150:
        return s.rstrip(" .")
    cut = s[:150]
    for sep in [",", ";", " "]:
        pos = cut.rfind(sep)
        if pos >= 60:
            cut = cut[:pos]
            break
    return cut.rstrip(" .")

# ----------------------------
# UI
# ----------------------------
def ui_single(model):
    st.subheader("Single Input")
    col1, col2 = st.columns(2)
    with col1:
        raw_text = st.text_area("Testo", height=180,
                                placeholder="Incolla qui il testo…")
        url = st.text_input("Website URL", placeholder="https://…")
    with col2:
        ean = st.text_input("EAN (solo cifre)", placeholder="Es. 8001234567890")
        confirm_ean = False
        if ean and re.fullmatch(r"\\d{8,14}", ean) and not raw_text.strip() and not url.strip():
            st.info(EAN_QUESTION)
            confirm_ean = st.button("Sì, procedi")

    if st.button("Build Description", type="primary") or (ean and confirm_ean):
        if raw_text.strip():
            source_text = raw_text.strip()
        elif url.strip():
            source_text = extract_main_text_from_url(url.strip())
        elif ean and re.fullmatch(r"\\d{8,14}", ean):
            if not confirm_ean:
                st.stop()
            fetched, _ = fetch_by_ean(ean)
            if not fetched:
                st.error("Nessuna informazione trovata per questo EAN nel web.")
                st.stop()
            source_text = fetched
        else:
            st.warning("Inserisci almeno una sorgente valida (testo, URL o EAN).")
            st.stop()

        with st.spinner("Calling Gemini…"):
            html_out = run_gemini_to_html(model, source_text)

        descr_txt = extract_descr_from_html(html_out)
        short_out = make_short_description(descr_txt)

        st.markdown("### #modifica testo (copia con un click)")
        st.code(html_out, language="html")
        st.markdown(ELEGANT_DIVIDER)
        st.markdown("### Descrizione breve (≤150 caratteri)")
        st.write(short_out)

def ui_batch(model):
    st.subheader("Batch via Excel")
    file = st.file_uploader("Upload Excel", type=["xlsx","xls"])
    if not file:
        return
    df = pd.read_excel(file)
    outputs = []
    for idx, row in df.iterrows():
        html_out = ""
        if isinstance(row.get('testo'), str) and row['testo'].strip():
            html_out = run_gemini_to_html(model, row['testo'].strip())
        elif isinstance(row.get('url'), str) and row['url'].strip():
            src = extract_main_text_from_url(str(row['url']).strip())
            html_out = run_gemini_to_html(model, src)
        elif pd.notna(row.get('ean')):
            ean_str = re.sub(r"\\D", "", str(row['ean']))
            fetched, _ = fetch_by_ean(ean_str)
            if fetched:
                html_out = run_gemini_to_html(model, fetched)
        if not html_out:
            continue
        descr_txt = extract_descr_from_html(html_out)
        short_out = make_short_description(descr_txt)
        outputs.append({'row': idx, 'html': html_out, 'short': short_out})

    st.success(f"Generated {len(outputs)} record(s).")
    if outputs:
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
        preset_api = get_api_key_from_env_or_secrets()
        api_key = st.text_input("Google API Key", type="password", value=preset_api or "")
        model_name = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
        st.caption("La chiave nei Secrets ha priorità; questo campo è un fallback.")

    if not api_key:
        st.info("Inserisci la Google API Key o configura i Secrets.")
        return

    try:
        model = get_gemini_model(api_key, model_name)
        st.session_state['model'] = model
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
