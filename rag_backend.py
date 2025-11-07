import os
import re
import warnings
import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer
from transformers import pipeline

warnings.filterwarnings("ignore")

# ============================
# 1. RUTAS
# ============================
DATA_PATH = "Car_Reviews_español.xlsx"
EMB_PATH = "car_reviews_embeddings.npy"

# ============================
# 2. CARGA DE DATOS
# ============================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"No se encontró el archivo {DATA_PATH}. "
        "Subilo al mismo repo/carpeta donde está app.py."
    )

df = pd.read_excel(DATA_PATH)
df = df.dropna(subset=["review_es"])

documents = [
    f"Vehículo: {row['Vehicle_Title']}. Comentario del usuario: {row['review_es']}. ¿Lo recomienda?: {row['Recommend']}."
    for _, row in df.iterrows()
]

# lo uso para saber rápido si un token existe en algún título
ALL_TITLES_LOW = df["Vehicle_Title"].str.lower()

print(f"✅ Dataset cargado: {len(df)} reseñas")

# ============================
# 3. EMBEDDINGS
# ============================
print("Cargando modelo de SentenceTransformer - all-MiniLM-L6-v2 ...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
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
print(f"Forma de los embeddings: {doc_embeddings.shape}")

# ============================
# 4. FAISS
# ============================
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)
print(f"Índice FAISS creado con {index.ntotal} vectores.")

# ============================
# 5. MODELO GENERATIVO
# ============================
print("Cargando modelo generativo (GPT-2 español)...")
generator = pipeline(
    task="text-generation",
    model="datificate/gpt2-small-spanish",
    device=-1
)
print("Modelo generativo cargado ✅")

# ============================
# 6. CONSTANTES
# ============================
KNOWN_BRANDS = ["toyota", "ford", "hyundai", "kia", "honda", "chevrolet", "chevy"]
KNOWN_MODELS = ["corolla", "transit", "cr-v", "crv", "tucson", "santa fe", "spin",
                "edge", "rio", "soul", "veracruz", "element"]

STOP_TOKENS = {
    "la", "el", "de", "del", "los", "las", "y", "sobre", "que",
    "una", "un", "al", "en", "para", "por", "con", "lo", "los", "las"
}

QUERY_EXPANSIONS = {
    "caja automática": ["transmisión", "transmision", "caja", "automatic"],
    "consumo": ["consumo", "mpg", "gasolina", "rendimiento de gasolina", "economía de combustible"],
    "ruido": ["ruido", "ruidoso", "cabina ruidosa", "sound", "noise"],
}

ASPECT_EXPANSIONS = {
    "consumo": ["consumo", "gasolina", "rendimiento", "mpg", "fuel", "economía", "economia"],
    "ruido": ["ruido", "ruidoso", "sonido", "cabina ruidosa"],
    "transmisión": ["transmisión", "transmision", "caja", "caja automática", "automatic"],
    "suspensión": ["suspensión", "suspension"],
    "aire": ["aire acondicionado", "ac", "a/c", "climatizador"],
    "asientos": ["asiento", "asientos", "confort", "comodidad"],
    "motor": ["motor", "potencia", "engine"],
    "espacio": ["espacio", "carga", "maletero", "baúl", "baul"],
}

# ============================
# 7. HELPERS
# ============================
def clean_token(t: str) -> str:
    # saca ¿ ? ! . , ; :
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

def filter_docs_by_aspect(retrieved_docs: list[str], query: str) -> list[str]:
    aspect_terms = get_aspect_terms(query)
    if not aspect_terms:
        return retrieved_docs
    filtered_chunks = []
    for doc in retrieved_docs:
        sentences = re.split(r"[\.!?]\s+", doc)
        for sent in sentences:
            if any(term in sent.lower() for term in aspect_terms):
                filtered_chunks.append(sent.strip() + ".")
    return filtered_chunks or retrieved_docs

def extract_years_from_docs(docs: list[str]) -> list[str]:
    years = set()
    for d in docs:
        found = re.findall(r"\b(?:19|20)\d{2}\b", d)
        for y in found:
            years.add(y)
    return sorted(list(years))

# ============================
# 8. RETRIEVAL
# ============================
def retrieve_documents(query: str, k: int = 3):
    """
    - si hay marca en la pregunta (ford) y además alguna palabra de la pregunta
      aparece en algún Vehicle_Title, filtramos por marca + esas palabras
      (esto captura "ford aspire")
    - si solo hay marca, filtramos por marca
    - si no, FAISS global
    """
    q_low = query.lower()
    raw_tokens = [t for t in re.split(r"\s+", q_low) if t]
    tokens = [clean_token(t) for t in raw_tokens if clean_token(t)]

    brands_in_query = [b for b in KNOWN_BRANDS if b in q_low]

    # tokens de la pregunta que de verdad existen en títulos (aspire, connect, cr-v...)
    model_like_tokens = []
    for t in tokens:
        if t in STOP_TOKENS:
            continue
        if t in KNOWN_BRANDS:
            continue
        if ALL_TITLES_LOW.str.contains(t).any():
            model_like_tokens.append(t)

    # 1) marca + token existente en títulos
    if brands_in_query and model_like_tokens:
        mask = pd.Series([True] * len(df))
        mask &= df["Vehicle_Title"].str.lower().str.contains("|".join(brands_in_query))
        for tok in model_like_tokens:
            mask &= df["Vehicle_Title"].str.lower().str.contains(tok)

        df_sub = df[mask]
        if not df_sub.empty:
            df_sub = df_sub.reset_index(drop=True)
            sub_docs = [
                f"Vehículo: {row['Vehicle_Title']}. Comentario del usuario: {row['review_es']}. ¿Lo recomienda?: {row['Recommend']}."
                for _, row in df_sub.iterrows()
            ]
            return sub_docs[:k]

    # 2) solo marca
    if brands_in_query:
        mask = df["Vehicle_Title"].str.lower().str.contains("|".join(brands_in_query))
        df_sub = df[mask]
        if not df_sub.empty:
            df_sub = df_sub.reset_index(drop=True)
            sub_docs = [
                f"Vehículo: {row['Vehicle_Title']}. Comentario del usuario: {row['review_es']}. ¿Lo recomienda?: {row['Recommend']}."
                for _, row in df_sub.iterrows()
            ]
            # faiss dentro del subset para quedarnos con los más parecidos
            sub_emb = embedding_model.encode(sub_docs, show_progress_bar=False)
            sub_emb = np.array(sub_emb, dtype="float32")
            sub_index = faiss.IndexFlatL2(dimension)
            sub_index.add(sub_emb)

            expanded_query = expand_query(query)
            q_emb = np.array(embedding_model.encode([expanded_query]), dtype="float32")
            dists, idxs = sub_index.search(q_emb, min(k, len(df_sub)))
            return [sub_docs[i] for i in idxs[0]]

    # 3) fallback global
    expanded_query = expand_query(query)
    q_emb = np.array(embedding_model.encode([expanded_query]), dtype="float32")
    dists, idxs = index.search(q_emb, k)
    return [documents[i] for i in idxs[0]]

# ============================
# 9. REDACTOR HUMANO
# ============================
def build_human_answer(query: str, docs: list[str]) -> str:
    aspect_terms = get_aspect_terms(query)
    comentarios = []
    vehiculos = set()
    positivos = 0
    negativos = 0

    for d in docs:
        if d.startswith("Vehículo:"):
            parte_vehiculo = d.split("Comentario del usuario:")[0]
            vehiculos.add(parte_vehiculo.replace("Vehículo:", "").strip().strip("."))
        if "Comentario del usuario:" in d:
            com = d.split("Comentario del usuario:")[1]
            com = com.split("¿Lo recomienda?")[0].strip()
        else:
            com = d.strip()
        comentarios.append(com)
        if "¿Lo recomienda?: Si" in d or "¿Lo recomienda?: Sí" in d:
            positivos += 1
        elif "¿Lo recomienda?: No" in d:
            negativos += 1

    if aspect_terms:
        comentarios = [c for c in comentarios if any(t in c.lower() for t in aspect_terms)] or comentarios

    resp = "Esto es lo que dicen las reseñas encontradas:\n\n"
    if vehiculos:
        resp += "Modelos con opiniones: " + ", ".join(vehiculos) + ".\n"
    if positivos or negativos:
        total = positivos + negativos
        resp += f"De {total} opiniones, {positivos} son positivas y {negativos} son negativas.\n"
    resp += "\nPrincipales comentarios:\n"
    for c in comentarios[:3]:
        resp += f"- {c}\n"

    return resp.strip()

# ============================
# 10. GENERACIÓN
# ============================
def generate_answer(query: str, retrieved_docs: list[str]) -> str:
    if not retrieved_docs:
        return "No tengo información suficiente para responder a esa pregunta."

    max_chars = 1200
    short_docs = [str(doc)[:max_chars] for doc in retrieved_docs]
    context = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(short_docs))

    prompt = (
        "Tienes opiniones de usuarios sobre vehículos (en español).\n"
        "Resume de manera natural y completa lo que dicen los usuarios sobre el tema consultado.\n"
        "No repitas el texto exacto, sintetiza y escribe una respuesta terminada con punto final.\n"
        "Si no hay información suficiente, responde literalmente:\n"
        "\"No tengo información suficiente para responder a esa pregunta.\"\n\n"
        f"Opiniones:\n{context}\n\n"
        f"Pregunta: {query}\n"
        "Respuesta:"
    )

    try:
        out = generator(
            prompt,
            max_new_tokens=350,
            num_return_sequences=1,
            temperature=0.4,
            top_p=0.9,
            return_full_text=False,
        )
        model_text = out[0]["generated_text"]
    except Exception:
        model_text = ""

    for token in ["Pregunta:", "Opiniones:", "Contexto:"]:
        if token in model_text:
            model_text = model_text.split(token)[0]
    model_text = " ".join(line.strip() for line in model_text.splitlines() if line.strip())

    truncated = not model_text.endswith((".", "?", "!", "”", "\"")) or len(model_text.split()) < 20
    too_similar = "comentario del usuario" in model_text.lower()

    if truncated or too_similar:
        return build_human_answer(query, retrieved_docs)

    return model_text.strip()

# ============================
# 11. PIPELINE FINAL
# ============================
def answer_pipeline(query: str) -> str:
    # 1) recuperamos como siempre (3 docs)
    docs = retrieve_documents(query, k=3)

    # 2) intentamos detectar el modelo exacto del primer doc
    modelo_detectado = None
    if docs:
        first = docs[0]
        if first.startswith("Vehículo:"):
            modelo_detectado = first.split("Comentario del usuario:")[0]
            modelo_detectado = modelo_detectado.replace("Vehículo:", "").strip().lower()

    # 3) si detectamos modelo, traemos TODAS las filas de ese modelo
    if modelo_detectado:
        # escapamos por las dudas (si hay parentesis, etc.)
        mask_all = df["Vehicle_Title"].str.lower().str.contains(re.escape(modelo_detectado))
        df_all = df[mask_all]
        if not df_all.empty:
            all_docs_same_model = [
                f"Vehículo: {row['Vehicle_Title']}. Comentario del usuario: {row['review_es']}. ¿Lo recomienda?: {row['Recommend']}."
                for _, row in df_all.iterrows()
            ]
            return generate_answer(query, all_docs_same_model)

    # 4) si no hubo modelo exacto, seguimos con la lógica de años
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

    # 5) respuesta normal
    return generate_answer(query, docs)





