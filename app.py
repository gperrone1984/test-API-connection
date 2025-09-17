# streamlit_app.py
# Requirements (install in your environment):
#   streamlit
#   google-generativeai>=0.7.0
#   pandas
#   requests
#   beautifulsoup4
#   lxml
#   readability-lxml
#   openpyxl

import os
import re
import io
import json
import zipfile
from typing import Dict, Tuple

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from readability import Document

try:
    import google.generativeai as genai
except Exception:
    genai = None

TITLE = "Product Description Builder (Gemini)"

from textwrap import dedent

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

\n"
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


ELEGANT_DIVIDER = """\n---\n"""

EAN_QUESTION = "Vuoi avere informazioni sul prodotto collegato a questo EAN?"

HEADERS = {"User-Agent": "Mozilla/5.0"}

LOWER_WORDS_IT = {"di","a","da","in","con","su","per","tra","fra","e","o","dei","degli","delle","del","della","dell'","agli","alle","al","allo","ai","lo","la","le","il","un","uno","una","ed","od"}


def to_capitalized_case(text: str) -> str:
    words = re.split(r"(\s+)", text.strip())
    out = []
    for w in words:
        if w.isspace():
            out.append(w)
            continue
        bare = re.sub(r"^[^\w]*|[^\w]*$", "", w)
        if bare.lower() in LOWER_WORDS_IT:
            out.append(bare.lower())
        else:
            out.append(bare.capitalize())
    return "".join(out)


def ensure_trailing_period(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    if s.endswith((".", "!", "?")):
        return s
    return s + '.'


def remove_inner_br(text: str) -> str:
    return re.sub(r"<\s*br\s*/?>", " ", text, flags=re.IGNORECASE)


def get_gemini_model(api_key: str, model_name: str = "gemini-1.5-flash"):
    if genai is None:
        raise RuntimeError("google-generativeai non è installato.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def run_gemini_to_html(model, source_text: str) -> str:
    """Invia a Gemini il prompt e ritorna SOLO l'HTML finale generato (no JSON)."""
    prompt = ITALIAN_PROMPT_PREAMBLE + "

#input
" + source_text
    resp = model.generate_content(prompt, request_options={"timeout": 60})
    text = getattr(resp, 'text', None) or str(resp)
    # Tieni solo i blocchi <p>...</p>
    parts = re.findall(r"<p[\s\S]*?</p>", text, flags=re.IGNORECASE)
    html = " ".join(parts).strip() if parts else text.strip()
    return html


def extract_ingredients_sections(soup: BeautifulSoup) -> str:
    labels = ["ingredienti", "inci", "composizione", "componenti"]
    texts = []
    for tag in soup.find_all(["h1","h2","h3","h4","h5","strong","b","p","span"]):
        t = (tag.get_text(" ") or "").strip().lower()
        if any(lbl in t for lbl in labels):
            nxts = []
            for sib in list(tag.next_siblings)[:5]:
                if getattr(sib, 'name', None) in ("ul","ol","p","div","span","table"):
                    nxts.append(sib)
            chunk = []
            for el in nxts:
                try:
                    chunk.append(el.get_text(" ").strip())
                except Exception:
                    continue
            if chunk:
                texts.append(" ".join(chunk))
    return " \n".join(texts)


def extract_main_text_from_url(url: str) -> str:
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


def duckduckgo_search_urls(query: str, max_results: int = 6) -> list:
    try:
        resp = requests.post("https://duckduckgo.com/html/", data={"q": query}, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        s = BeautifulSoup(resp.text, 'lxml')
        urls = []
        for a in s.select("a.result__a"):
            href = a.get('href', '')
            if href.startswith("http"):
                urls.append(href)
            if len(urls) >= max_results:
                break
        return urls
    except Exception:
        return []


def fetch_by_ean(ean: str) -> Tuple[str, Dict[str, str]]:
    """Cerca il prodotto nel web (no Open*Facts). Ritorna testo aggregato e metadati."""
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
    return ("

".join(texts), {"source": "web"}), {"source": "web", "urls": urls})


def build_final_html(blocks: Dict[str, str]) -> str:
    titolo = to_capitalized_case(remove_inner_br(blocks.get('titolo','').strip()))
    descr = remove_inner_br(blocks.get('descrizione_generale','').strip())
    modo  = remove_inner_br(blocks.get('modo_uso','').strip())
    ingr  = remove_inner_br(blocks.get('ingredienti','').strip())
    avv   = remove_inner_br(blocks.get('avvertenze','').strip())
    form  = remove_inner_br(blocks.get('formato','').strip())

    if not modo:
        modo = "Per il corretto modo d'uso si prega di fare riferimento alla confezione"
    if not ingr:
        ingr = "Per la lista completa degli ingredienti si prega di fare riferimento alla confezione"

    if titolo:
        titolo = to_capitalized_case(titolo)
    if ingr and not ingr.startswith("Per la lista completa"):
        ingr = to_capitalized_case(ingr)

    descr = ensure_trailing_period(descr) if descr else descr
    modo  = ensure_trailing_period(modo) if modo else modo
    ingr  = ensure_trailing_period(ingr) if ingr else ingr
    avv   = ensure_trailing_period(avv) if avv else avv
    form  = ensure_trailing_period(form) if form else form

    parts = []
    parts.append(f"<p><strong>{titolo}</strong><br>")
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
        s = re.sub(r"\s\S*$", "", s).strip()
    return s.rstrip(" .")

def extract_descr_from_html(html: str) -> str:
    try:
        m = re.search(r"<p>\s*<strong>.*?</strong><br>\s*(.*?)</p>", html, flags=re.IGNORECASE|re.DOTALL)
        if not m:
            return ""
        frag = m.group(1)
        txt = BeautifulSoup(frag, "lxml").get_text(" ")
        return re.sub(r"\s+", " ", txt).strip()
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


def ui_single(model):
    st.subheader("Single Input")
    col1, col2 = st.columns(2)
    with col1:
        raw_text = st.text_area("Text (Italian preferred)", height=180)
        url = st.text_input("Website URL")
    with col2:
        ean = st.text_input("EAN (digits only)")
        confirm_ean = False
        if ean and re.fullmatch(r"\d{8,14}", ean) and not raw_text.strip() and not url.strip():
            st.info(EAN_QUESTION)
            confirm_ean = st.button("Sì, procedi")

    if st.button("Build Description", type="primary") or (ean and confirm_ean):
        source_text = ""
        if raw_text.strip():
            source_text = raw_text.strip()
        elif url.strip():
            source_text = extract_main_text_from_url(url.strip())
        elif ean and re.fullmatch(r"\d{8,14}", ean):
            if not confirm_ean:
                st.stop()
            fetched, _ = fetch_by_ean(ean)
            if not fetched:
                st.error("Nessuna informazione trovata per questo EAN.")
                st.stop()
            source_text = f"[FONTE: EAN {ean}] " + fetched
        else:
            st.warning("Inserisci almeno una sorgente valida.")
            st.stop()

        with st.spinner("Calling Gemini…"):
            blocks = run_gemini_extraction(model, source_text)

        html_out = build_final_html(blocks)
        short_out = trim_short_description(blocks.get("descrizione_breve", "") or blocks.get("descrizione_generale", ""))

        st.markdown("### #modifica testo")
        st.code(html_out, language="html")
        st.markdown(ELEGANT_DIVIDER)
        st.markdown("### Descrizione breve")
        st.write(short_out)


def ui_batch(model):
    st.subheader("Batch via Excel")
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
            blocks = {k: str(row.get(k) or '') for k in ["titolo","descrizione_generale","modo_uso","ingredienti","avvertenze","formato"]}
            blocks['descrizione_breve'] = ''
        html_out = build_final_html(blocks)
        short_out = trim_short_description(blocks.get("descrizione_breve", "") or blocks.get("descrizione_generale", ""))
        outputs.append({'row': idx,'html': html_out,'short': short_out})
    st.success(f"Generated {len(outputs)} record(s).")
    if outputs:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for o in outputs:
                base = f"row_{o['row']}"
                zf.writestr(f"{
