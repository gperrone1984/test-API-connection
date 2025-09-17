# streamlit_app.py
# Requirements (put these in requirements.txt for Streamlit Cloud):
#   streamlit
#   google-generativeai>=0.7.0
#   pandas
#   requests
#   beautifulsoup4
#   lxml
#   readability-lxml
#   openpyxl
#
# Secrets (Streamlit Cloud → Settings → Secrets):
#   GOOGLE_API_KEY = "la_tua_chiave"
#
# Funzioni principali:
# - Input: Testo / URL / EAN (con domanda di consenso obbligatoria).
# - URL: copia ESATTA del contenuto del sito (no parafrasi), organizzato nelle sezioni.
# - EAN: prova Open*Facts; se vuoto, cerca sul web e concatena snippet pertinenti, poi Gemini struttura/parafrasa.
# - Output: formato HTML conforme alle regole (#modifica testo) + Descrizione breve (≤150 caratteri, senza punto finale).

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

try:
    import google.generativeai as genai
except Exception:
    genai = None

# ----------------------------
# Costanti & Helpers
# ----------------------------
TITLE = "Product Description Builder (Gemini)"

ITALIAN_PROMPT_PREAMBLE = (
    "Dobbiamo creare la descrizione di un prodotto. Siamo di Redcare Pharmacy, una farmacia online.\n"
    "Rimani fedele a quello che ti ho scritto.\n\n"
    "#imput\n"
    "- Se incollo un testo utilizza solo le informazioni che ci sono nel testo senza inventare nulla.\n"
    "- Se incollo un sito web ricordati che ho i diritti per utilizzare i contenuti. Quindi non inventare nulla. Utilizza solo il sito che ti do. Se non trovi le informazioni non inventare nulla.\n"
    "- Se incollo un EAN di un prodotto NON SCRIVERE NULLA: Prima chiedimi se voglio cercare le informazioni del prototto collegate all`EAN. La domanda da fare è ESCLUSIVAMENTE: \"Vuoi avere informazioni sul prodotto collegato a questo EAN?\"\n"
    "- Trova tutte le informazioni che riesci online collegato SOLO all´EAN che ti fornisco. Il prodotto deve essere effettivamente quello collegato all´EAN.\n"
    "- Quando crei una descrizione da un EAN per la descrizione generale parafrasa quello che prendi da altri siti, non posso copiare esattamente i contenuti da un sito  a meno che non sia quello del produttore.\n\n"
    "#modifica testo\n"
    "- Nei prossimi passaggi Il testo inserito non deve essere modificato, solo copiato. Non aggiungere o inventare nulla.\n"
    "- Fai un passaggio per volta.\n"
    "- Il titolo è la parte prima di 'descrizione' o 'Indicazioni'. Il testo deve essere in Capitalized Case. aggiungi <p><strong> prima del titolo e </strong><br> dopo del titolo. Elimina i <br> all'interno del testo\n"
    "- Sotto il titolo va la descrizione generale del prodotto. Direttamente sotto il titolo, senza scrivere descrizione. Alla fine aggiungi </p>\n"
    "- In Modo d'uso: devi prendere il testo di Modalità d'uso. Se la modalità d'uso/modo dúso manca utilizza la seguente frase: 'Per il corretto modo d'uso si prega di fare riferimento alla confezione'. Aggiungi <p><strong> prima di Modo d'uso: e </strong><br> alla fine. Alla fine di tutto metti </p>\n"
    "- In Ingredienti: vanno Ingredienti o componenti. Converti il testo degli ingredienti in capitalized case.  Gli ingredienti devono essere in una forma impersonale. Se gli ingredienti mancano inserire la frase: Per la lista completa degli ingredienti si prega di fare riferimento alla confezione.  Aggiungi <p><strong> prima di Ingredienti: e </strong> <br> alla fine. Alla fine di tutto metti </p>. Converti il testo degli ingredienti in capitalized case. Non mettere in capitalized case la frase: Per la lista completa degli ingredienti si prega di fare riferimento alla confezione.\n"
    "- In Avvertenze: vanno Avvertenze. Aggiungi <p><strong> prima di Avvertenze: e </strong> <br> alla fine. Alla fine di tutto metti </p>. Se le Avvertenze non ci sono non scrivere nulla.\n"
    "- Per i dispositivi medici se presenti aggiungi il Formato, Aggiungi <p><strong> prima di Formato: e </strong> <br> alla fine. Alla fine di tutto metti </p>. Se il Formato non c´é non scrivere nulla.\n\n"
    "#Output richiesto\n"
    "Restituisci un JSON con le seguenti chiavi (usa stringhe vuote se mancanti):\n"
    "{\"titolo\": str, \"descrizione_generale\": str, \"modo_uso\": str, \"ingredienti\": str, \"avvertenze\": str, \"formato\": str, \"descrizione_breve\": str}"
)

ELEGANT_DIVIDER = """\n---\n"""

EAN_QUESTION = "Vuoi avere informazioni sul prodotto collegato a questo EAN?"

LOWER_WORDS_IT = {
    "di","a","da","in","con","su","per","tra","fra","e","o","dei","degli","delle","del","della","dell'","agli","alle","al","allo","ai","lo","la","le","il","un","uno","una","ed","od"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

# ----------------------------
# Utilities
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


def ensure_trailing_period(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s.endswith(('.', '!', '?')):
        return s
    return s + '.'


def remove_inner_br(text: str) -> str:
    return re.sub(r"<\s*br\s*/?>", " ", text, flags=re.IGNORECASE)


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


def run_gemini_extraction(model, source_text: str) -> Dict[str, str]:
    # Modalità speciale per URL: copia esatta (niente parafrasi) tramite marker opzionale
    site_mode = False
    marker = "[DA_SITO: COPIA_ESATTA]"
    if source_text.startswith(marker):
        site_mode = True
        source_text = source_text[len(marker):].lstrip()

    extra = ""
    if site_mode:
        extra = (
            "\n\n[ISTRUZIONI PER SITO]\n"
            "- NON parafrasare il contenuto della descrizione generale: copia il testo esattamente come appare nella sorgente (salvo le trasformazioni richieste in #modifica testo).\n"
            "- Mappa il contenuto nelle sezioni richieste: titolo, descrizione_generale, modo_uso, ingredienti, avvertenze, formato.\n"
            "- Se una sezione non è presente nel sito, lasciala vuota (applicheremo i fallback dove previsti).\n"
        )

    prompt = ITALIAN_PROMPT_PREAMBLE + extra + "\n\n#testo\n" + source_text
    resp = model.generate_content(prompt, request_options={"timeout": 60})
    text = getattr(resp, 'text', None) or str(resp)

    m = re.search(r"\{[\s\S]*\}$", text.strip())
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
        keys = ["titolo","descrizione_generale","modo_uso","ingredienti","avvertenze","formato","descrizione_breve"]
        for k in keys:
            data.setdefault(k, "")
        return {k: (data.get(k) or "").strip() for k in keys}
    except Exception:
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
# Source acquisition (URL / EAN) + Web Search
# ----------------------------

def extract_ingredients_sections(soup: BeautifulSoup) -> str:
    """Tenta di estrarre blocchi Ingredienti/INCI/Composizione dal DOM."""
    labels = ["ingredienti", "inci", "composizione", "componenti"]
    texts: List[str] = []
    for tag in soup.find_all(["h1","h2","h3","h4","h5","strong","b","p","span"]):
        t = (tag.get_text(" ") or "").strip().lower()
        if any(lbl in t for lbl in labels):
            nxts = []
            if tag.name not in ("strong","b"):
                nxts.append(tag)
            for sib in list(tag.next_siblings)[:6]:
                if getattr(sib, 'name', None) in ("ul","ol","p","div","span","table"):
                    nxts.append(sib)
            par = tag.parent
            if par and par != tag and par.name in ("p","div","section","li"):
                nxts.append(par)
            chunk = []
            for el in nxts:
                try:
                    chunk.append(el.get_text(" ").strip())
                except Exception:
                    continue
            block = " ".join(chunk).strip()
            if block:
                texts.append(block)
    out, seen = [], set()
    for t in texts:
        if t and t not in seen:
            out.append(t); seen.add(t)
    return " \n".join(out)[:4000]


def extract_main_text_from_url(url: str) -> str:
    """Estrae testo leggibile e prova a preservare sezioni come Ingredienti/INCI."""
    try:
        r = requests.get(url, timeout=25, headers=HEADERS)
        r.raise_for_status()
        doc = Document(r.text)
        html = doc.summary()
        soup = BeautifulSoup(html, 'lxml')
        for tag in soup(['script','style','noscript']):
            tag.decompose()
        ingr = extract_ingredients_sections(soup)
        text = soup.get_text(" ")
        text = re.sub(r"\s+", " ", text).strip()
        if ingr and 'ingredient' not in text.lower():
            text = text + " \n\nIngredienti: " + ingr
        return text[:24000]
    except Exception as e:
        return f"[ERRORE estrazione sito]: {e}"


# Helpers to copy exact content from websites into fields (no paraphrase)

def find_first_text(soup: BeautifulSoup, selectors: List[str]) -> str:
    for sel in selectors:
        node = soup.select_one(sel)
        if node:
            txt = node.get_text(" ").strip()
            if txt:
                return txt
    return ""


def extract_section_by_label(soup: BeautifulSoup, labels: List[str]) -> str:
    for tag in soup.find_all(["h1","h2","h3","h4","h5","strong","b","p","span"]):
        t = (tag.get_text(" ") or "").strip().lower()
        if any(lbl in t for lbl in labels):
            chunks = []
            if tag.name not in ("strong","b"):
                chunks.append(tag.get_text(" ").strip())
            for sib in list(tag.next_siblings)[:6]:
                if getattr(sib, 'name', None) in ("ul","ol","p","div","span","table"):
                    chunks.append(sib.get_text(" ").strip())
            t_all = " ".join([c for c in chunks if c]).strip()
            if t_all:
                for lbl in labels:
                    t_all = re.sub(rf"(?i){re.escape(lbl)}\s*[:|-]*\s*", "", t_all).strip()
                return t_all
    return ""


def extract_fields_from_url(url: str) -> Dict[str, str]:
    try:
        r = requests.get(url, timeout=25, headers=HEADERS)
        r.raise_for_status()
        doc = Document(r.text)
        html = doc.summary()
        soup = BeautifulSoup(html, "lxml")
        for t in soup(["script","style","noscript"]):
            t.decompose()
        titolo = find_first_text(soup, ["h1", "header h1", "h1.product-title"]) or find_first_text(soup, ["title"]) or ""
        descr = find_first_text(soup, ["article p", ".product-description p", ".content p", "main p", "p"]) or ""
        modo  = extract_section_by_label(soup, ["modo d'uso", "modalità d'uso"]) or ""
        ingr  = extract_section_by_label(soup, ["ingredienti", "inci", "composizione", "componenti"]) or ""
        avv   = extract_section_by_label(soup, ["avvertenze", "precauzioni", "attenzione"]) or ""
        form  = extract_section_by_label(soup, ["formato", "confezione", "contenuto"]) or ""
        return {
            "titolo": titolo,
            "descrizione_generale": descr,
            "modo_uso": modo,
            "ingredienti": ingr,
            "avvertenze": avv,
            "formato": form,
            "descrizione_breve": ""
        }
    except Exception as e:
        return {
            "titolo": "",
            "descrizione_generale": f"[ERRORE estrazione sito]: {e}",
            "modo_uso": "",
            "ingredienti": "",
            "avvertenze": "",
            "formato": "",
            "descrizione_breve": ""
        }


def duckduckgo_search_urls(query: str, max_results: int = 6) -> List[str]:
    try:
        resp = requests.post("https://duckduckgo.com/html/", data={"q": query}, headers=HEADERS, timeout=20)
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


def fetch_by_ean(ean: str) -> Tuple[str, Dict[str, Any]]:
    """Cerca dati prodotto per EAN: Open*Facts → (se vuoto) web search + snippet."""
    endpoints = [
        f"https://world.openfoodfacts.org/api/v0/product/{ean}.json",
        f"https://world.openbeautyfacts.org/api/v0/product/{ean}.json",
        f"https://world.openpetfoodfacts.org/api/v0/product/{ean}.json",
    ]
    for url in endpoints:
        try:
            r = requests.get(url, timeout=15, headers=HEADERS)
            if r.status_code == 200:
                data = r.json()
                if data.get('status') == 1 and 'product' in data:
                    p = data['product']
                    name = p.get('product_name') or p.get('generic_name') or ''
                    brand = p.get('brands', '')
                    qty = p.get('quantity', '')
                    ingredients = p.get('ingredients_text_it') or p.get('ingredients_text') or ''
                    bits = [b for b in [name, brand, qty, ingredients] if b]
                    joined = "; ".join(bits)
                    if joined:
                        return joined, {"source": "openfacts", "raw": data}
        except Exception:
            continue

    # Web search fallback
    snippets, seen = [], set()
    urls = duckduckgo_search_urls(f"{ean} sito ufficiale") or duckduckgo_search_urls(ean)
    for u in urls[:6]:
        if u in seen:
            continue
        seen.add(u)
        try:
            txt = extract_main_text_from_url(u)
            if ean in txt or len(txt) > 300:
                snippets.append(f"[URL] {u}\n{txt[:6000]}")
        except Exception:
            continue
    return ("\n\n".join(snippets), {"source": "web", "urls": urls})


# ----------------------------
# Formattazione finale (#modifica testo)
# ----------------------------

def build_final_html(blocks: Dict[str, str]) -> str:
    titolo = to_capitalized_case(remove_inner_br(blocks.get('titolo','').strip()))
    descr  = remove_inner_br(blocks.get('descrizione_generale','').strip())
    modo   = remove_inner_br(blocks.get('modo_uso','').strip())
    ingr   = remove_inner_br(blocks.get('ingredienti','').strip())
    avv    = remove_inner_br(blocks.get('avvertenze','').strip())
    form   = remove_inner_br(blocks.get('formato','').strip())

    if not modo:
        modo = "Per il corretto modo d'uso si prega di fare riferimento alla confezione"
    if not ingr:
        ingr = "Per la lista completa degli ingredienti si prega di fare riferimento alla confezione"

    if titolo:
        titolo = to_capitalized_case(titolo)
    if ingr and not ingr.startswith("Per la lista completa"):
        ingr = to_capitalized_case(ingr)

    descr = ensure_trailing_period(descr) if descr else descr
    modo  = ensure_trailing_period(modo)  if modo  else modo
    if ingr:
        ingr = ensure_trailing_period(ingr)
    avv   = ensure_trailing_period(avv)   if avv   else avv
    form  = ensure_trailing_period(form)  if form  else form

    parts = []
    parts.append(f"<p><strong>{titolo}</strong><br> {descr}</p>")
    parts.append(f"<p><strong>Modo d'uso: </strong><br> {modo}</p>")
    parts.append(f"<p><strong>Ingredienti: </strong><br> {ingr}</p>")
    if avv:
        parts.append(f"<p><strong>Avvertenze: </strong><br> {avv}</p>")
    if form:
        parts.append(f"<p><strong>Formato: </strong><br> {form}</p>")
    return " ".join(parts)


def trim_short_description(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) > 150:
        s = s[:150]
        s = re.sub(r"\s\S*$", "", s).strip()
    return s.rstrip(" .")


def normalize_short(short_from_model: str, fallback: str) -> str:
    val = (short_from_model or "").strip()
    if val.lower() in ("undefined", "null", "n/a", "-") or not val:
        val = (fallback or "").strip()
    return trim_short_description(val)


# ----------------------------
# UI
# ----------------------------

def ui_single(model):
    st.subheader("Single Input")
    col1, col2 = st.columns(2)
    with col1:
        raw_text = st.text_area("Testo", height=180, placeholder="Incolla qui il testo da cui prendere le informazioni…")
        url = st.text_input("Website URL", placeholder="https://…")
    with col2:
        ean = st.text_input("EAN (solo cifre)", placeholder="Es. 8001234567890")
        confirm_ean = False
        if ean and re.fullmatch(r"\d{8,14}", ean) and not raw_text.strip() and not url.strip():
            st.info(EAN_QUESTION)
            confirm_ean = st.button("Sì, procedi")

    if st.button("Build Description", type="primary") or (ean and confirm_ean):
        if raw_text.strip():
            # Testo incollato → Gemini per strutturare rispettando le regole
            with st.spinner("Calling Gemini…"):
                blocks = run_gemini_extraction(model, raw_text.strip())

        elif url.strip():
            # URL → copia ESATTA (no parafrasi): estrai campi e NON usare Gemini
            blocks = extract_fields_from_url(url.strip())

        elif ean and re.fullmatch(r"\d{8,14}", ean):
            if not confirm_ean:
                st.stop()
            fetched, _ = fetch_by_ean(ean)
            if not fetched:
                st.error("Nessuna informazione pubblica trovata per questo EAN. Incolla l'URL del produttore o di un sito autorizzato.")
                st.stop()
            with st.spinner("Calling Gemini…"):
                blocks = run_gemini_extraction(model, f"[FONTE: EAN {ean}] {fetched}")
        else:
            st.warning("Inserisci almeno una sorgente valida (testo, URL o EAN).")
            st.stop()

        html_out = build_final_html(blocks)
        short_out = normalize_short(blocks.get("descrizione_breve"), blocks.get("descrizione_generale"))

        st.markdown("### #modifica testo (copia con un click)")
        st.code(html_out, language="html")
        st.markdown(ELEGANT_DIVIDER)
        st.markdown("### Descrizione breve (≤150 caratteri)")
        st.write(short_out)


def ui_batch(model):
    st.subheader("Batch via Excel")
    st.write("Carica un Excel con colonne (facoltative): **titolo, descrizione_generale, modo_uso, ingredienti, avvertenze, formato, testo, url, ean**.")

    file = st.file_uploader("Upload Excel", type=["xlsx","xls"])
    if not file:
        return

    df = pd.read_excel(file)
    outputs = []

    for idx, row in df.iterrows():
        src_text = ""
        blocks: Dict[str, str]

        if isinstance(row.get('testo'), str) and row['testo'].strip():
            src_text = row['testo'].strip()
            blocks = run_gemini_extraction(model, src_text)
        elif isinstance(row.get('url'), str) and row['url'].strip():
            # URL → copia esatta
            blocks = extract_fields_from_url(str(row['url']).strip())
        elif pd.notna(row.get('ean')):
            ean_str = re.sub(r"\D", "", str(row['ean']))
            fetched, _ = fetch_by_ean(ean_str)
            if fetched:
                blocks = run_gemini_extraction(model, f"[FONTE: EAN {ean_str}] {fetched}")
            else:
                blocks = {k: '' for k in ["titolo","descrizione_generale","modo_uso","ingredienti","avvertenze","formato","descrizione_breve"]}
        else:
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
        short_out = normalize_short(blocks.get("descrizione_breve"), blocks.get("descrizione_generale"))

        outputs.append({
            'row': idx,
            'html': html_out,
            'short': short_out
        })

    st.success(f"Generati {len(outputs)} record.")
    if outputs:
        st.markdown("#### Preview (primi 3)")
        for o in outputs[:3]:
            st.code(o['html'], language='html')
            st.write("**Descrizione breve:** ", o['short'])
            st.markdown(ELEGANT_DIVIDER)

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
