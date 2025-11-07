import os
import re
import warnings
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# =====================================================
# 1. RUTAS
# =====================================================
DATA_PATH = "Car_Reviews_español.xlsx"
EMB_PATH = "car_reviews_embeddings.npy"

# =====================================================
# 2. CARGA DE DATOS
# =====================================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"No se encontró el archivo {DATA_PATH}. Subilo al mismo repo/carpeta donde está app.py."
    )

df = pd.read_excel(DATA_PATH)
df = df.dropna(subset=["review_es"])

documents = [
    f"Vehículo: {row['Vehicle_Title']}. Comentario del usuario: {row['review_es']}. ¿Lo recomienda?: {row['Recommend']}."
    for _, row in df.iterrows()
]

ALL_TITLES_LOW = df["Vehicle_Title"].str.lower()

print(f"✅ Dataset cargado: {len(df)} reseñas")

# =====================================================
# 3. EMBEDDINGS + FAISS
# =====================================================
print("Cargando modelo de SentenceTransformer - all-MiniLM-L6-v2 ...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dim = 384
print("Modelo cargado ✅")

if os.path.exists(EMB_PATH):
    doc_embeddings = np.load(EMB_PATH)
    print(f"Embeddings cargados desde {EMB_PATH}")
else:
    print("Generando embeddings (primera vez)...")
    doc_embeddings = embedding_model.encode(documents, show_progress_bar=False)
    doc_embeddings = np.array(doc_embeddings, dtype="float32")
    try:
        np.save(EMB_PATH, doc_embeddings)
    except Exception:
        pass
    print("✅ Embeddings generados y guardados")

doc_embeddings = np.array(doc_embeddings, dtype="float32")

index = faiss.IndexFlatL2(dim)
index.add(doc_embeddings)
print(f"Índice FAISS creado con {index.ntotal} vectores.")


# =====================================================
# 4. LISTAS / EXPANSIONES
# =====================================================
KNOWN_BRANDS = ["toyota", "ford", "hyundai", "kia", "honda", "chevrolet", "chevy"]

STOP_TOKENS = {
    "la", "el", "de", "del", "los", "las", "y", "sobre", "que",
    "una", "un", "al", "en", "para", "por", "con", "lo"
}

QUERY_EXPANSIONS = {
    "caja automática": ["transmisión", "transmision", "caja", "automatic"],
    "consumo": [
        "consumo", "combustible", "consumo de combustible",
        "mpg", "gasolina", "rendimiento de gasolina",
        "economía de combustible", "kilometraje", "millas", "km"
    ],
    "ruido": ["ruido", "ruidoso", "cabina ruidosa", "sound", "noise"],
}

ASPECT_EXPANSIONS = {
    "consumo": [
        "consumo", "combustible", "consumo de combustible",
        "gasolina", "rendimiento", "mpg", "economía", "economia"
    ],
    "ruido": ["ruido", "ruidoso", "sonido", "cabina ruidosa"],
    "transmisión": ["transmisión", "transmision", "caja", "caja automática", "automatic"],
    "motor": ["motor", "potencia", "engine"],
    "espacio": ["espacio", "carga", "maletero", "baúl", "baul"],
    "aire": ["aire acondicionado", "ac", "a/c", "climatizador"],
}


# =====================================================
# 5. HELPERS
# =====================================================
def clean_token(t: str) -> str:
    return re.sub(r'^[\?\!\.\,\;\:\¿\¡]+|[\?\!\.\,\;\:\¿\¡]+$', '', t)

def expand_query(query: str) -> str:
    q = query.lower()
    extra = []
    for key, vals in QUERY_EXPANSIONS.items():
        if key in q:
            extra.extend(vals)
    return query + " " + " ".join(extra) if extra else query

def has_query_year(query: str) -> str | None:
    m = re.search(r"\b(19|20)\d{2}\b", query)
    return m.group(0) if m else None

def context_has_year(context: str, year: str) -> bool:
    return year in context

def get_aspect_terms(query: str):
    q = query.lower()
    terms = set()
    for key, vals in ASPECT_EXPANSIONS.items():
        if key in q:
            terms.update(vals)
    return list(terms)


def rerank_by_aspect(docs: list[str], query: str) -> list[str]:
    """Reordena los docs poniendo arriba los que más mencionan el aspecto."""
    aspect_terms = get_aspect_terms(query)
    if not aspect_terms:
        return docs

    scored = []
    for d in docs:
        text_low = d.lower()
        score = sum(text_low.count(term) for term in aspect_terms)
        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored[0][0] == 0:
        return [d for _, d in scored]
    return [d for _, d in scored]


def filter_docs_by_aspect(retrieved_docs: list[str], query: str) -> list[str]:
    aspect_terms = get_aspect_terms(query)
    if not aspect_terms:
        return retrieved_docs

    filtered = []
    for doc in retrieved_docs:
        sentences = re.split(r"[\.!?]\s+", doc)
        for sent in sentences:
            if any(term in sent.lower() for term in aspect_terms):
                filtered.append(sent.strip() + ".")

    return filtered or retrieved_docs


def extract_years_from_docs(docs: list[str]) -> list[str]:
    years = set()
    for d in docs:
        found = re.findall(r"\b(?:19|20)\d{2}\b", d)
        for y in found:
            years.add(y)
    return sorted(list(years))


# =====================================================
# 6. RETRIEVAL
# =====================================================
def retrieve_documents(query: str, k: int = 3):
    q_low = query.lower()
    tokens = [clean_token(t) for t in q_low.split() if clean_token(t)]
    brands_in_query = [b for b in KNOWN_BRANDS if b in q_low]

    model_like_tokens = []
    for t in tokens:
        if t in STOP_TOKENS or t in KNOWN_BRANDS:
            continue
        if ALL_TITLES_LOW.str.contains(t).any():
            model_like_tokens.append(t)

    if brands_in_query and model_like_tokens:
        mask = pd.Series([True] * len(df))
        mask &= df["Vehicle_Title"].str.lower().str.contains("|".join(brands_in_query))
        for tok in model_like_tokens:
            mask &= df["Vehicle_Title"].str.lower().str.contains(tok)

        df_sub = df[mask]
        if not df_sub.empty:
            df_sub = df_sub.reset_index(drop=True)
            return [
                f"Vehículo: {row['Vehicle_Title']}. Comentario del usuario: {row['review_es']}. ¿Lo recomienda?: {row['Recommend']}."
                for _, row in df_sub.head(k).iterrows()
            ]

    if brands_in_query:
        mask = df["Vehicle_Title"].str.lower().str.contains("|".join(brands_in_query))
        df_sub = df[mask]
        if not df_sub.empty:
            sub_docs = [
                f"Vehículo: {row['Vehicle_Title']}. Comentario del usuario: {row['review_es']}. ¿Lo recomienda?: {row['Recommend']}."
                for _, row in df_sub.iterrows()
            ]
            sub_emb = embedding_model.encode(sub_docs, show_progress_bar=False)
            sub_emb = np.array(sub_emb, dtype="float32")
            sub_index = faiss.IndexFlatL2(dim)
            sub_index.add(sub_emb)

            expanded_query = expand_query(query)
            q_emb = np.array(embedding_model.encode([expanded_query]), dtype="float32")
            dists, idxs = sub_index.search(q_emb, min(k, len(sub_docs)))
            return [sub_docs[i] for i in idxs[0]]

    expanded_query = expand_query(query)
    q_emb = np.array(embedding_model.encode([expanded_query]), dtype="float32")
    dists, idxs = index.search(q_emb, k)
    return [documents[i] for i in idxs[0]]


# =====================================================
# 7. CONSTRUCTOR DE RESPUESTA
# =====================================================
def build_human_answer(query: str, docs: list[str]) -> str:
    aspect_terms = get_aspect_terms(query)
    vehiculos = set()
    comentarios = []
    pos = 0
    neg = 0

    for d in docs:
        if d.startswith("Vehículo:"):
            vh = d.split("Comentario del usuario:")[0]
            vh = vh.replace("Vehículo:", "").strip().strip(".")
            vehiculos.add(vh)
        if "Comentario del usuario:" in d:
            com = d.split("Comentario del usuario:")[1]
            com = com.split("¿Lo recomienda?")[0].strip()
        else:
            com = d.strip()
        comentarios.append(com)

        if "¿Lo recomienda?: Si" in d or "¿Lo recomienda?: Sí" in d:
            pos += 1
        elif "¿Lo recomienda?: No" in d:
            neg += 1

    if aspect_terms:
        comentarios_aspecto = [
            c for c in comentarios if any(t in c.lower() for t in aspect_terms)
        ]
        if comentarios_aspecto:
            comentarios = comentarios_aspecto

    if not comentarios:
        return "No tengo información suficiente para responder a esa pregunta."

    resp = "Esto es lo que dicen las reseñas encontradas:\n\n"
    if vehiculos:
        resp += "Modelos con opiniones: " + ", ".join(vehiculos) + ".\n"
    if pos or neg:
        total = pos + neg
        resp += f"De {total} opiniones, {pos} son positivas y {neg} son negativas.\n"
    resp += "\nPrincipales comentarios:\n"
    for c in comentarios[:3]:
        resp += f"- {c}\n"

    return resp.strip()


# =====================================================
# 8. PIPELINE FINAL
# =====================================================
def answer_pipeline(query: str) -> str:
    docs = retrieve_documents(query, k=4)
    docs = rerank_by_aspect(docs, query)

    year = has_query_year(query)
    ctx = "\n\n".join(docs)
    if year and not context_has_year(ctx, year):
        yrs = extract_years_from_docs(docs)
        if yrs:
            return (
                f"No encuentro reseñas del {year}.\n"
                f"Pero sí hay opiniones de estos años: {', '.join(yrs)}.\n"
                "¿Querés que te cuente sobre alguno de esos?"
            )
        else:
            return (
                f"No encuentro reseñas del {year}. "
                "Solo hay comentarios de otros años o modelos."
            )

    aspect_docs = filter_docs_by_aspect(docs, query)
    return build_human_answer(query, aspect_docs)


def retrieve_for_ui(query: str, k: int = 4):
    return retrieve_documents(query, k)
