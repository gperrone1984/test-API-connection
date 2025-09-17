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

# ----------------------------
# Config & Costanti
# ----------------------------
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

# Prompt completo (solo output HTML)
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
# Helpers di formattazione / casing
# ----------------------------
def to_capitalized_case(text: str) -> str:
    def fix_token(tok: str) -> str:
        if not tok:
            return tok
        if re.fullmatch(r"[A-Z0-9]{2,}", tok):
            return tok
        base = re.sub(r"^([\W_]*)(.*?)([\W_]*)$", r"\1:::\2:::\3", tok)
        pre, core, post = base.split(":::")
        if core.lower() in LOWER_WORDS_IT:
            return f"{pre}{core.lower()}{post}"
        return f"{pre}{core[:1].upper()}{core[1:].lower()}{post}"
    tokens = re.split(r"(\s+)", text.strip())
    return ''.join([fix_token(t) if not t.isspace() else t for t in tokens])

def remove_inner_br(text: str) -> str:
    return re.sub(r"<\s*br\s*/?>", " ", text, flags=re.IGNORECASE)

def is_all_caps(s: str) -> bool:
    letters = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]", "", s or "")
    return bool(letters) and letters.upper() == letters

def sentence_case_from_all_caps(s: str) -> str:
    if not s:
        return s
    low = s.lower()
    def cap_sentences(t: str) -> str:
        out = []
        cap_next = True
        for ch in t:
            if cap_next and ch.isalpha():
                out.append(ch.upper()); cap_next = False
            else:
                out.append(ch)
            if ch in ".!?":
                cap_next = True
        return "".join(out)
    t = cap_sentences(low)
    # ripristina sigle 2-4 lettere (best-effort)
    def restore_acronyms(m):
        return m.group(0).upper()
    t = re.sub(r"\b([a-z]{2,4})\b", restore_acronyms, t)
    return t

def to_capitalized_case_ingredients(s: str) -> str:
    if not s:
        return s
    # spezza su punti elenco e virgole
    parts = re.split(r"\s*[•\u2022\-\–\—]|,\s*", s)
    cleaned = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        tokens = p.split()
        fixed = []
        for tok in tokens:
            if len(tok) <= 4 and tok.isupper():
                fixed.append(tok)  # lascia INCI/sigle corte
            else:
                fixed.append(to_capitalized_case(tok))
        cleaned.append(" ".join(fixed))
    return ", ".join(cleaned)

def ensure_trailing_period(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    return s if s.endswith(('.', '!', '?')) else s + '.'

# ----------------------------
# Gemini setup & call
# ----------------------------
def get_api_key_from_env_or_secrets() -> Optional[str]:
    if 'GOOGLE_API_KEY' in st.secrets:
        return st.secrets['GOOGLE_API_KEY']
    return os.getenv('GOOGLE_API_KEY')

def get_gemini_model(api_key: str, model_name: str = "gemini-1.5-flash"):
    if genai is None:
        raise RuntimeError("google-generativeai non è installato.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def run_gemini_to_html(model, source_text: str) -> str:
    prompt = f"{ITALIAN_PROMPT_PREAMBLE}\n\n{source_text}"
    resp = model.generate_content(prompt, request_options={"timeout": 60})
    text = getattr(resp, 'text', None) or str(resp)
    # raccogli solo blocchi <p>...</p>
    parts = re.findall(r"<p[\s\S]*?</p>", text, flags=re.IGNORECASE)
    html = " ".join(parts).strip() if parts else text.strip()
    return html

# ----------------------------
# Ricerca web per EAN (no Open*Facts)
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
        return re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        return f"[ERRORE estrazione sito]: {e}"

def fetch_by_ean(ean: str) -> Tuple[str, Dict[str, str]]:
    queries = [f'"{ean}" sito ufficiale', str(ean)]
    urls: List[str] = []
    for q in queries:
        res = duckduckgo_search_urls(q)
        for u in res:
            if u not in urls:
                urls.append(u)
        if len(urls) >= 8:
            break
    texts: List[str] = []
    for u in urls[:8]:
        try:
            t = extract_main_text_from_url(u)
            if len(t) > 300:
                texts.append(t)
        except Exception:
            continue
    return ("\n\n".join(texts), {"source": "web"})

# ----------------------------
# Generazione descrizione breve
# ----------------------------
def extract_descr_from_html(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
        p = soup.find("p")
        if not p:
            return ""
        text = p.get_text(" ").strip()
        strong = p.find("strong")
        if strong:
            strong_text = strong.get_text(" ").strip()
            text = text.replace(strong_text, "", 1).strip()
        return re.sub(r"\s+", " ", text).strip(" :-")
    except Exception:
        return ""

def make_short_description(descr: str) -> str:
    s = re.sub(r"\s+", " ", (descr or "").strip())
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
# Costruzione HTML finale
# ----------------------------
def build_final_html(blocks: Dict[str, str]) -> str:
    titolo = remove_inner_br((blocks.get('titolo') or '').strip())
    descr  = remove_inner_br((blocks.get('descrizione_generale') or '').strip())
    modo   = remove_inner_br((blocks.get('modo_uso') or '').strip())
    ingr   = remove_inner_br((blocks.get('ingredienti') or '').strip())
    avv    = remove_inner_br((blocks.get('avvertenze') or '').strip())
    form   = remove_inner_br((blocks.get('formato') or '').strip())

    # Titolo in Capitalized Case
    if titolo:
        titolo = to_capitalized_case(titolo)

    # Normalizza sezioni se ricevute ALL CAPS
    for name in ["descr", "modo", "avv", "form"]:
        val = locals()[name]
        if is_all_caps(val):
            locals()[name] = sentence_case_from_all_caps(val)

    # Fallback Modo d'uso
    if not modo:
        modo = "Per il corretto modo d'uso si prega di fare riferimento alla confezione"
    # Ingredienti
    if not ingr:
        ingr = "Per la lista completa degli ingredienti si prega di fare riferimento alla confezione"
    else:
        ingr = to_capitalized_case_ingredients(ingr)

    # Punti finali
    if descr: descr = ensure_trailing_period(descr)
    if modo:  modo  = ensure_trailing_period(modo)
    if ingr:  ingr  = ensure_trailing_period(ingr)
    if avv:   avv   = ensure_trailing_period(avv)
    if form:  form  = ensure_trailing_period(form)

    parts = []
    parts.append(f"<p><strong>{titolo}</strong><br> {descr}</p>")
    parts.append(f"<p><strong>Modo d'uso: </strong><br> {modo}</p>")
    parts.append(f"<p><strong>Ingredienti: </strong><br> {ingr}</p>")
    if avv:
        parts.append(f"<p><strong>Avvertenze: </strong><br> {avv}</p>")
    if form:
        parts.append(f"<p><strong>Formato: </strong><br> {form}</p>")
    return " ".join(parts)

# ----------------------------
# UI
# ----------------------------
def ui_single(model):
    st.subheader("Single Input")
    col1, col2 = st.columns(2)
    with col1:
        raw_text = st.text_area("Testo", height=180, placeholder="Incolla qui il testo…")
        url = st.text_input("Website URL", placeholder="https://…")
    with col2:
        ean = st.text_input("EAN (solo cifre)", placeholder="Es. 8001234567890")
        confirm_ean = False
        # Mostra sempre la domanda ESATTA se c'è EAN valido e nessun altro input
        if ean and re.fullmatch(r"\d{8,14}", ean) and not raw_text.strip() and not url.strip():
            st.info('Vuoi avere informazioni sul prodotto collegato a questo EAN?')
            confirm_ean = st.checkbox("Sì, procedi", value=False)

    build = st.button("Build Description", type="primary")

    if build:
        if raw_text.strip():
            source_text = raw_text.strip()

        elif url.strip():
            source_text = extract_main_text_from_url(url.strip())

        elif ean and re.fullmatch(r"\d{8,14}", ean):
            if not confirm_ean:
                st.error('Vuoi avere informazioni sul prodotto collegato a questo EAN?')
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

        # Descrizione breve
        descr_txt = extract_descr_from_html(html_out)
        short_out = make_short_description(descr_txt)

        st.markdown("### #modifica testo (copia con un click)")
        st.code(html_out, language="html")
        st.markdown(ELEGANT_DIVIDER)
        st.markdown("### Descrizione breve (≤150 caratteri)")
        st.write(short_out)

def ui_batch(model):
    st.subheader("Batch via Excel")
    st.write("Carica un Excel con colonne (almeno una tra testo/url/ean): "
             "**titolo, descrizione_generale, modo_uso, ingredienti, avvertenze, formato, testo, url, ean**.")
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
            ean_str = re.sub(r"\D", "", str(row['ean']))
            fetched, _ = fetch_by_ean(ean_str)
            if fetched:
                html_out = run_gemini_to_html(model, fetched)
        if not html_out:
            continue
        descr_txt = extract_descr_from_html(html_out)
        short_out = make_short_description(descr_txt)
        outputs.append({'row': idx, 'html': html_out, 'short': short_out})

    st.success(f"Generati {len(outputs)} record.")
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
