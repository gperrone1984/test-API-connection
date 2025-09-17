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

# ----------------------------
# Prompt (modello → solo HTML finale)
# ----------------------------
ITALIAN_PROMPT_PREAMBLE = dedent("""
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
# Helpers formattazione / casing
# ----------------------------
def to_capitalized_case(text: str) -> str:
    """
    Capitalized Case per titoli/parole:
    - lascia intatte sigle 2+ lettere/numero tutte maiuscole (INCI, SPF, 3D, etc.)
    - parole funzione in minuscolo (di, a, da, in, con, su, per, tra, fra, e, o, del...)
    - preserva punteggiatura adiacente; niente backreference tipo \1\2\3
    """
    if not text:
        return ""
    tokens = re.split(r"(\s+)", text.strip())
    out = []
    for tok in tokens:
        if tok.isspace():
            out.append(tok)
            continue

        m = re.match(r"^([\W_]*)(.*?)([\W_]*)$", tok, flags=re.UNICODE | re.DOTALL)
        if not m:
            out.append(tok)
            continue

        pre, core, post = m.group(1), m.group(2), m.group(3)

        if re.fullmatch(r"[A-Z0-9]{2,}", core):
            fixed_core = core  # sigla → lascia
        else:
            # separa eventuali trattini/apostrofi interni
            parts = re.split(r"([\-’'\\/])", core)
            new_parts = []
            for p in parts:
                if re.fullmatch(r"[\-’'\\/]", p):
                    new_parts.append(p)
                else:
                    if p.lower() in LOWER_WORDS_IT:
                        new_parts.append(p.lower())
                    else:
                        new_parts.append(p[:1].upper() + p[1:].lower() if p else p)
            fixed_core = "".join(new_parts)

        out.append(f"{pre}{fixed_core}{post}")
    return "".join(out)

def remove_inner_br(text: str) -> str:
    return re.sub(r"<\s*br\s*/?>", " ", text or "", flags=re.IGNORECASE)

def ensure_trailing_period(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    return s if s.endswith(('.', '!', '?')) else s + '.'

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
                fixed.append(tok)  # INCI/sigle corte → lascia
            else:
                fixed.append(to_capitalized_case(tok))
        cleaned.append(" ".join(fixed))
    return ", ".join(cleaned)

def normalize_sentence_case(text: str) -> str:
    """Forza sempre il sentence case; preserva sigle di 2–4 lettere."""
    if not text:
        return ""
    s = re.sub(r"\s+", " ", text.strip()).lower()
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
    return s.strip()

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
    mini = dedent(f"""
    Crea una descrizione breve (massimo 150 caratteri, senza punto finale) basata sul seguente testo.
    Non copiare letteralmente frasi intere: sintetizza in modo naturale.
    Testo:
    {descr}
    """)
    try:
        resp = model.generate_content(mini, request_options={"timeout": 30})
        s = (getattr(resp, "text", None) or str(resp)).strip()
        s = re.sub(r"\s+", " ", s).split("\n")[0].strip()
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
# Ricerca EAN: DuckDuckGo /lite + /html (no Google API)
# ----------------------------
DDG_HEADERS = {"User-Agent": HEADERS["User-Agent"]}

SEARCH_BLACKLIST = {
    "amazon.", "ebay.", "aliexpress.", "pinterest.", "facebook.", "instagram.",
    "twitter.", "x.com", "tiktok.", "youtube.", "openfoodfacts.", "openbeautyfacts.",
    "idealo.", "trovaprezzi.", "kelkoo.", "google.", "bing.", "yahoo."
}

def is_blacklisted(url: str) -> bool:
    host = url.lower()
    return any(b in host for b in SEARCH_BLACKLIST)

def duckduckgo_search_urls(query: str, max_results: int = 8) -> List[str]:
    """Tenta prima /lite (GET), poi /html (POST)."""
    urls: List[str] = []
    try:
        r = requests.get("https://duckduckgo.com/lite/", params={"q": query}, headers=DDG_HEADERS, timeout=20)
        r.raise_for_status()
        s = BeautifulSoup(r.text, "lxml")
        for a in s.select("a[href]"):
            href = a.get("href", "")
            if href.startswith("http") and not is_blacklisted(href):
                urls.append(href)
            if len(urls) >= max_results:
                break
    except Exception:
        pass
    if len(urls) < max_results:
        try:
            resp = requests.post("https://duckduckgo.com/html/", data={"q": query}, headers=DDG_HEADERS, timeout=20)
            resp.raise_for_status()
            s = BeautifulSoup(resp.text, 'lxml')
            for a in s.select("a.result__a[href]"):
                href = a.get('href', '')
                if href.startswith("http") and not is_blacklisted(href):
                    if href not in urls:
                        urls.append(href)
                if len(urls) >= max_results:
                    break
        except Exception:
            pass
    return urls

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
    Affidabile = pagina contiene l'EAN o JSON-LD Product con GTIN/ EAN uguale.
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

        # JSON-LD Product
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

        # Readability text
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

        # match EAN nella pagina
        if ean and (ean in html_full or ean in text):
            reliable = True

        return text, reliable
    except Exception:
        return "", False

def fetch_by_ean(ean: str) -> Tuple[str, Dict[str, Any]]:
    """Ricerca con più query e criteri di affidabilità (EAN nel testo o JSON-LD GTIN)."""
    queries = [
        f"\"{ean}\"",
        f"{ean} gtin13",
        f"{ean} gtin",
        f"{ean} barcode",
        f"{ean} scheda tecnica",
        f"{ean} ingredienti",
        f"{ean} INCI",
        f"{ean} pdf",
        f"{ean} sito ufficiale",
    ]
    seen, urls = set(), []
    for q in queries:
        for u in duckduckgo_search_urls(q, max_results=10):
            if u not in seen:
                urls.append(u); seen.add(u)
            if len(urls) >= 24:
                break
        if len(urls) >= 24:
            break

    snippets, good_hits, debug = [], 0, []
    for u in urls:
        text, reliable = extract_main_text_from_url(u, ean=ean)
        if not text:
            debug.append((u, "no_text"))
            continue
        if reliable:
            good_hits += 1
            snippets.append(text)
            debug.append((u, "reliable"))
        else:
            if len([s for s in snippets if s.startswith("[FALLBACK]")]) < 2:
                snippets.append("[FALLBACK]\n" + text)
                debug.append((u, "fallback"))
        if good_hits >= 3:
            break

    return ("\n\n".join(snippets), {"urls": urls, "good_hits": good_hits, "debug": debug})

# ----------------------------
# Parsing HTML di Gemini → blocchi → normalizzazione → HTML finale
# ----------------------------
def parse_html_blocks(html: str) -> Dict[str, str]:
    """
    Estrae: titolo; descrizione_generale (testo del primo <p> al netto del <strong>);
    e le sezioni Modo d'uso / Ingredienti / Avvertenze / Formato dai successivi <p>.
    """
    out = {"titolo": "", "descrizione_generale": "", "modo_uso": "", "ingredienti": "", "avvertenze": "", "formato": ""}

    soup = BeautifulSoup(html, "lxml")
    ps = soup.find_all("p")
    if not ps:
        return out

    # Paragrafo 1: titolo + descrizione
    p0 = ps[0]
    strong = p0.find("strong")
    if strong:
        out["titolo"] = strong.get_text(" ").strip()
    text_p0 = p0.get_text(" ").strip()
    if out["titolo"]:
        text_p0 = text_p0.replace(out["titolo"], "", 1).strip()
    text_p0 = text_p0.lstrip(" :-")
    out["descrizione_generale"] = re.sub(r"\s+", " ", text_p0).strip()

    # Altri paragrafi
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
    # Accetta solo quantità reali (ml, g, l, capsule, compresse, pz, bustine, ecc.)
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

    # Formato: quantità plausibili solo
    form = clean_format(form)

    # Normalizza sempre in sentence case (evita maiuscolo urlato)
    for name in ["descr", "modo", "avv", "form"]:
        locals()[name] = normalize_sentence_case(locals()[name])

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
# Estrarre descrizione generale dall'HTML finale (per descrizione breve)
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
        manual_urls = st.text_area("URL aggiuntivi (uno per riga, es. sito produttore)", height=80, placeholder="https://…\nhttps://…")
    with col2:
        ean_input = st.text_input("EAN (solo cifre)", placeholder="Es. 8001234567890")
        # normalizza EAN (solo cifre)
        ean = re.sub(r"\D", "", ean_input or "")
        confirm_ean = False
        if ean and re.fullmatch(r"\d{8,14}", ean) and not raw_text.strip() and not url.strip():
            st.info(EAN_QUESTION)
            confirm_ean = st.checkbox("Sì, procedi", value=False)
        debug_mode = st.checkbox("Mostra debug ricerca", value=False)

    build = st.button("Build Description", type="primary")

    if build:
        # 1) Sorgente: testo, URL, EAN (in questo ordine)
        if raw_text.strip():
            source_text = raw_text.strip()

        elif url.strip():
            txt, _ = extract_main_text_from_url(url.strip(), ean=None)
            source_text = txt or " "

        elif ean and re.fullmatch(r"\d{8,14}", ean):
            if not confirm_ean:
                st.error(EAN_QUESTION)
                st.stop()
            fetched, meta = fetch_by_ean(ean)

            # opzionale: arricchisci con URL manuali (produttore ecc.)
            extra_texts = []
            if manual_urls.strip():
                for line in manual_urls.strip().splitlines():
                    u = line.strip()
                    if u.startswith("http"):
                        t, _ = extract_main_text_from_url(u, ean=ean)
                        if t:
                            extra_texts.append(t)
            if extra_texts:
                fetched = "\n\n".join(extra_texts + [fetched])

            if not fetched or meta.get("good_hits", 0) == 0:
                st.error("Nessuna informazione trovata per questo EAN nel web.")
                if debug_mode:
                    st.markdown("### Debug ricerca EAN")
                    st.write(meta)
                st.stop()
            source_text = fetched
            if debug_mode:
                st.markdown("### Debug ricerca EAN")
                st.write(meta.get("debug", []))

        else:
            st.warning("Inserisci almeno una sorgente valida (testo, URL o EAN).")
            st.stop()

        # 2) Chiedi a Gemini l'HTML come da prompt
        with st.spinner("Calling Gemini…"):
            html_raw = run_gemini_to_html(model, source_text)

        # 3) Parsa/normalizza e ricostruisci nel formato esatto
        blocks = parse_html_blocks(html_raw)
        html_out = build_final_html(blocks)

        # 4) Descrizione breve da zero sulla base della descrizione generale
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
            if re.fullmatch(r"\d{8,14}", ean_str):
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
