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

# Prompt completo (Gemini può rispondere in HTML; poi lo normalizziamo)
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
    def restore_acronyms(m):
        return m.group(0).upper()
    t = re.sub(r"\b([a-z]{2,4})\b", restore_acronyms, t)
    return t

def normalize_sentence_case(text: str) -> str:
    """Forza sempre il sentence case (evita output urlato di Gemini)."""
    if not text:
        return ""
    s = text.lower()
    out = []
    cap_next = True
    for ch in s:
        if cap_next and ch.isalpha():
            out.append(ch.upper()); cap_next = False
        else:
            out.append(ch)
        if ch in ".!?":
            cap_next = True
    s = "".join(out)
    s = re.sub(r"\b([a-z]{2,4})\b", lambda m: m.group(1).upper(), s)  # ripristina sigle corte
    return s

def to_capitalized_case_ingredients(s: str) -> str:
    if not s:
        return s
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
    parts = re.findall(r"<p[\s\S]*?</p>", text, flags=re.IGNORECASE)
    html = " ".join(parts).strip() if parts else text.strip()
    return html

def run_gemini_shortdesc(model, descr: str) -> str:
    """Crea una descrizione breve da zero (≤150 caratteri, senza punto finale)."""
    mini = dedent(f"""\
    Crea una descrizione breve (massimo 150 caratteri, senza punto finale) basata sul seguente testo.
    Non copiare letteralmente frasi intere: sintetizza in modo naturale.
    Testo:
    {descr}
    """)
    try:
        resp = model.generate_content(mini, request_options={"timeout": 30})
        s = (getattr(resp, "text", None) or str(resp)).strip()
        s = re.sub(r"\s+", " ", s)
        s = s.split("\n")[0].strip()
        if len(s) > 150:
            s = s[:150]
            for sep in [",", ";", " "]:
                pos = s.rfind(sep)
                if pos >= 60:
                    s = s[:pos]
                    break
        return s.rstrip(" .")
    except Exception:
        s = re.sub(r"\s+", " ", (descr or "").strip())
        return s[:150].rstrip(" .")

# ----------------------------
# EAN SEARCH (NO GOOGLE API) — DuckDuckGo HTML + robust page extraction
# ----------------------------
DDG_HEADERS = {
    "User-Agent": HEADERS["User-Agent"]
}

SEARCH_BLACKLIST = {
    "amazon.", "ebay.", "aliexpress.", "pinterest.", "facebook.", "instagram.",
    "twitter.", "x.com", "tiktok.", "youtube.", "openfoodfacts.", "openbeautyfacts.",
    "idealo.", "trovaprezzi.", "kelkoo.", "google.", "bing.", "yahoo."
}

def is_blacklisted(url: str) -> bool:
    host = url.lower()
    return any(b in host for b in SEARCH_BLACKLIST)

def duckduckgo_search_urls(query: str, max_results: int = 8) -> List[str]:
    try:
        resp = requests.post("https://duckduckgo.com/html/",
                             data={"q": query}, headers=DDG_HEADERS, timeout=20)
        resp.raise_for_status()
        s = BeautifulSoup(resp.text, 'lxml')
        urls: List[str] = []
        for a in s.select("a.result__a"):
            href = a.get('href', '')
            if href and href.startswith("http") and not is_blacklisted(href):
                urls.append(href)
            if len(urls) >= max_results:
                break
        return urls
    except Exception:
        return []

def extract_jsonld_product(soup: BeautifulSoup) -> List[dict]:
    out = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            raw = script.string or ""
            if not raw.strip():
                continue
            data = json.loads(raw)
        except Exception:
            continue
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                t = str(item.get("@type", "")).lower()
                if t in {"product"}:
                    out.append(item)
    return out

def extract_main_text_from_url(url: str, ean: Optional[str] = None) -> Tuple[str, bool]:
    """
    Ritorna (testo_estratto, match_affidabile_con_ean).
    - Usa: <title>, meta description, JSON-LD Product (gtin/ean), e testo 'readable' (Readability).
    - 'affidabile' = pagina contiene l'EAN (html/testo) oppure JSON-LD con GTIN uguale all'EAN.
    """
    try:
        r = requests.get(url, timeout=25, headers=HEADERS)
        r.raise_for_status()
        html_full = r.text
        soup_full = BeautifulSoup(html_full, "lxml")

        # Title + meta description
        title = (soup_full.title.get_text(" ").strip() if soup_full.title else "")
        meta_desc = ""
        md = soup_full.find("meta", attrs={"name": "description"}) or soup_full.find("meta", attrs={"property": "og:description"})
        if md and md.get("content"):
            meta_desc = md["content"].strip()

        # JSON-LD Product (match GTIN/EAN)
        reliable = False
        jsonlds = extract_jsonld_product(soup_full)
        jsonld_bits = []
        if jsonlds:
            for item in jsonlds:
                gtins = []
                for k in ("gtin13", "gtin", "gtin12", "gtin14", "ean"):
                    v = item.get(k)
                    if v:
                        gtins.append(str(v))
                name = str(item.get("name") or "")
                desc = str(item.get("description") or "")
                if ean and any(ean in g for g in gtins):
                    reliable = True
                if name or desc:
                    jsonld_bits.append(f"Nome: {name}\nDescrizione: {desc}")

        # Readability main text
        doc = Document(html_full)
        html = doc.summary()
        soup = BeautifulSoup(html, 'lxml')
        for tag in soup(['script','style','noscript']):
            tag.decompose()
        text_readable = re.sub(r"\s+", " ", soup.get_text(" ")).strip()

        parts = []
        if title: parts.append(title)
        if meta_desc: parts.append(meta_desc)
        if jsonld_bits: parts.append("\n".join(jsonld_bits))
        if text_readable: parts.append(text_readable)
        text = "\n\n".join(p for p in parts if p)[:12000]

        # match stringa EAN nella pagina (html o testo)
        if ean and (ean in html_full or ean in text):
            reliable = True

        return text, reliable
    except Exception:
        return "", False

def fetch_by_ean(ean: str) -> Tuple[str, Dict[str, Any]]:
    """
    Ricerca robusta con DuckDuckGo HTML.
    - Prova più query (EAN puro, +gtin, +barcode, +scheda tecnica, +ingredienti/INCI).
    - Tieni le pagine 'affidabili' (contengono EAN o GTIN uguale).
    - Ritorna un testo aggregato e metadati (conteggio hit affidabili).
    """
    queries = [
        f"\"{ean}\"",
        f"{ean} gtin",
        f"{ean} gtin13",
        f"{ean} barcode",
        f"{ean} scheda tecnica",
        f"{ean} ingredienti",
        f"{ean} INCI",
    ]
    seen = set()
    urls: List[str] = []
    for q in queries:
        for u in duckduckgo_search_urls(q, max_results=8):
            if u not in seen:
                urls.append(u); seen.add(u)
            if len(urls) >= 20:
                break
        if len(urls) >= 20:
            break

    snippets: List[str] = []
    good_hits = 0
    for u in urls:
        text, reliable = extract_main_text_from_url(u, ean=ean)
        if not text:
            continue
        if reliable:
            good_hits += 1
            snippets.append(text)
        else:
            # prendi al massimo 2 fallback non affidabili per contesto
            if len([s for s in snippets if s.startswith("[FALLBACK]")]) < 2:
                snippets.append("[FALLBACK]\n" + text)
        if good_hits >= 3:  # sufficiente materiale 'buono'
            break

    return ("\n\n".join(snippets), {"urls": urls, "good_hits": good_hits})

# ----------------------------
# Parsing HTML di Gemini → blocchi → normalizzazione → HTML finale
# ----------------------------
def parse_html_blocks(html: str) -> Dict[str, str]:
    """
    Estrae: titolo, descrizione_generale (primo <p> dopo <strong><br>), e
    Modo d'uso / Ingredienti / Avvertenze / Formato dagli altri <p>.
    """
    out = {"titolo": "", "descrizione_generale": "", "modo_uso": "", "ingredienti": "", "avvertenze": "", "formato": ""}

    soup = BeautifulSoup(html, "lxml")
    ps = soup.find_all("p")
    if not ps:
        return out

    # Primo paragrafo: titolo + descrizione
    p0 = ps[0]
    strong = p0.find("strong")
    if strong:
        out["titolo"] = strong.get_text(" ").strip()
    text_p0 = p0.get_text(" ").strip()
    if out["titolo"]:
        text_p0 = text_p0.replace(out["titolo"], "", 1).strip()
    text_p0 = text_p0.lstrip(" :-")
    out["descrizione_generale"] = re.sub(r"\s+", " ", text_p0).strip()

    # Altri paragrafi con label
    for p in ps[1:]:
        label = ""
        strong = p.find("strong")
        if strong:
            label = strong.get_text(" ").strip().lower()
        content = p.get_text(" ").strip()
        if strong:
            content = content.replace(strong.get_text(" ").strip(), "", 1).strip()
        content = content.lstrip(": -").strip()
        if "modo d'uso" in label or "modalità d'uso" in label:
            out["modo_uso"] = content
        elif "ingredienti" in label or "inci" in label or "composizione" in label:
            out["ingredienti"] = content
        elif "avvertenze" in label or "precauzioni" in label or "attenzione" in label:
            out["avvertenze"] = content
        elif "formato" in label or "confezione" in label or "contenuto" in label:
            out["formato"] = content

    return out

def clean_format(value: str) -> str:
    v = re.sub(r"\s+", " ", (value or "").strip())
    # accetta solo quantità reali (ml, g, l, capsule, compresse, pz, bustine, ecc.)
    if re.search(r"\b\d+(?:[.,]\d+)?\s?(ml|g|kg|l|capsule|compresse|pz|bustine|sachets|tavolette)\b", v, re.IGNORECASE):
        return ensure_trailing_period(v)
    return ""  # scarta COD/SKU ecc.

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

    # Fallback Modo d'uso / Ingredienti
    if not modo:
        modo = "Per il corretto modo d'uso si prega di fare riferimento alla confezione"
    if not ingr:
        ingr = "Per la lista completa degli ingredienti si prega di fare riferimento alla confezione"
    else:
        ingr = to_capitalized_case_ingredients(ingr)

    # Formato: accetta solo quantità plausibili
    form = clean_format(form)

    # Normalizza SEMPRE in sentence case (evita maiuscolo urlato)
    for name in ["descr", "modo", "avv", "form"]:
        val = locals()[name]
        locals()[name] = normalize_sentence_case(val)

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
        text = re.sub(r"^\s*[:-]\s*", "", text).strip()
        return re.sub(r"\s+", " ", text)
    except Exception:
        return ""

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
        if ean and re.fullmatch(r"\d{8,14}", ean) and not raw_text.strip() and not url.strip():
            st.info('Vuoi avere informazioni sul prodotto collegato a questo EAN?')
            confirm_ean = st.checkbox("Sì, procedi", value=False)

    build = st.button("Build Description", type="primary")

    if build:
        if raw_text.strip():
            source_text = raw_text.strip()

        elif url.strip():
            txt, _ = extract_main_text_from_url(url.strip(), ean=None)
            source_text = txt or " "

        elif ean and re.fullmatch(r"\d{8,14}", ean):
            if not confirm_ean:
                st.error('Vuoi avere informazioni sul prodotto collegato a questo EAN?')
                st.stop()
            fetched, meta = fetch_by_ean(ean)
            if not fetched or meta.get("good_hits", 0) == 0:
                st.error("Nessuna informazione trovata per questo EAN nel web.")
                st.stop()
            source_text = fetched

        else:
            st.warning("Inserisci almeno una sorgente valida (testo, URL o EAN).")
            st.stop()

        # 1) Chiedi a Gemini l'HTML come da prompt
        with st.spinner("Calling Gemini…"):
            html_raw = run_gemini_to_html(model, source_text)

        # 2) Parsa/normalizza e ricostruisci nel formato esatto
        blocks = parse_html_blocks(html_raw)
        html_out = build_final_html(blocks)

        # 3) Descrizione breve da zero sulla base della descrizione generale
        descr_for_short = blocks.get("descrizione_generale", "")
        short_out = run_gemini_shortdesc(model, descr_for_short)

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
        src = ""
        if isinstance(row.get('testo'), str) and row['testo'].strip():
            src = row['testo'].strip()
        elif isinstance(row.get('url'), str) and row['url'].strip():
            src, _ = extract_main_text_from_url(str(row['url']).strip(), ean=None)
        elif pd.notna(row.get('ean')):
            ean_str = re.sub(r"\D", "", str(row['ean']))
            fetched, meta = fetch_by_ean(ean_str)
            if meta.get("good_hits", 0) > 0:
                src = fetched

        if not src:
            continue

        html_raw = run_gemini_to_html(model, src)
        blocks = parse_html_blocks(html_raw)
        html_out = build_final_html(blocks)
        short_out = run_gemini_shortdesc(model, blocks.get("descrizione_generale", ""))

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
