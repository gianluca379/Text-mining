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
# Rutas relativas (para GitHub / Streamlit Cloud)
# ============================
DATA_PATH = "Car_Reviews_español.xlsx"
EMB_PATH = "car_reviews_embeddings.npy"

# ============================
# 1. Carga de datos
# ============================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"No se encontró el archivo {DATA_PATH}. "
        "Subilo al mismo repo/carpeta donde está app.py."
    )

df = pd.read_excel(DATA_PATH)
df = df.dropna(subset=["review_es"])

# armamos los textos base
documents = [
    f"Vehículo: {row['Vehicle_Title']}. Comentario del usuario: {row['review_es']}. ¿Lo recomienda?: {row['Recommend']}."
    for _, row in df.iterrows()
]

print(f"✅ Dataset cargado: {len(df)} reseñas")

# ============================
# 2. Modelo de embeddings
# ============================
print("Cargando modelo de SentenceTransformer - all-MiniLM-L6-v2 ...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
print("Modelo cargado ✅")

# ============================
# 3. Embeddings (cargar o generar)
# ============================
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
# 5. Modelo generativo (lo mantenemos como opcional)
# ============================
print("Cargando modelo generativo (GPT-2 español)...")
generator = pipeline(
    task="text-generation",
    model="datificate/gpt2-small-spanish",
    device=-1
)
print("Modelo generativo cargado ✅")

# ============================
# 6. Listas / expansiones
# ============================
KNOWN_BRANDS = ["toyota", "ford", "hyundai", "kia", "honda", "chevrolet", "chevy"]
KNOWN_MODELS = ["corolla", "transit", "cr-v", "crv", "tucson", "santa fe", "spin",
                "edge", "rio", "soul", "veracruz", "element"]

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
# 7. Helpers
# ============================
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
# 8. Recuperación
# ============================
def retrieve_documents(query: str, k: int = 2):
    q_low = query.lower()
    brands_in_query = [b for b in KNOWN_BRANDS if b in q_low]
    models_in_query = [m for m in KNOWN_MODELS if m in q_low]

    # filtrado por marca / modelo
    if brands_in_query or models_in_query:
        mask = pd.Series([True] * len(df))
        if brands_in_query:
            mask = mask & df["Vehicle_Title"].str.lower().str.contains("|".join(brands_in_query))
        if models_in_query:
            mask = mask & df["Vehicle_Title"].str.lower().str.contains("|".join(models_in_query))
        df_sub = df[mask]
        if not df_sub.empty:
            sub_indices = df_sub.index.to_list()
            sub_embeddings = doc_embeddings[sub_indices]
            sub_index = faiss.IndexFlatL2(dimension)
            sub_index.add(sub_embeddings)

            expanded_query = expand_query(query)
            q_emb = embedding_model.encode([expanded_query])
            q_emb = np.array(q_emb, dtype="float32")
            dists, idxs = sub_index.search(q_emb, min(k, len(sub_indices)))

            retrieved = []
            for pos in idxs[0]:
                real_idx = sub_indices[pos]
                retrieved.append(documents[real_idx])
            return retrieved

    # búsqueda global
    expanded_query = expand_query(query)
    q_emb = embedding_model.encode([expanded_query])
    q_emb = np.array(q_emb, dtype="float32")
    dists, idxs = index.search(q_emb, k)
    return [documents[i] for i in idxs[0]]

# ============================
# 9. Redactor "humano" sin LLM
# ============================
def build_human_answer(query: str, docs: list[str]) -> str:
    """
    Construye una respuesta legible a partir de las reseñas recuperadas.
    No usa el modelo generativo, solo las reseñas.
    """
    aspect_terms = get_aspect_terms(query)
    comentarios = []
    vehiculos = set()
    positivos = 0
    negativos = 0

    for d in docs:
        # vehículo
        if d.startswith("Vehículo:"):
            parte_vehiculo = d.split("Comentario del usuario:")[0]
            vehiculos.add(parte_vehiculo.replace("Vehículo:", "").strip().strip("."))

        # comentario
        if "Comentario del usuario:" in d:
            com = d.split("Comentario del usuario:")[1]
            com = com.split("¿Lo recomienda?")[0].strip()
        else:
            com = d.strip()
        comentarios.append(com)

        # recomendación
        if "¿Lo recomienda?: Si" in d or "¿Lo recomienda?: Sí" in d:
            positivos += 1
        elif "¿Lo recomienda?: No" in d:
            negativos += 1

    # filtramos por aspecto si corresponde
    if aspect_terms:
        comentarios = [c for c in comentarios if any(t in c.lower() for t in aspect_terms)] or comentarios

    # armamos texto
    resp = "Esto es lo que dicen las reseñas que encontré sobre eso:\n\n"

    if vehiculos:
        resp += "Modelos de los que hay opiniones: " + ", ".join(vehiculos) + ".\n"

    if positivos or negativos:
        total = positivos + negativos
        resp += f"De {total} opiniones, {positivos} son positivas y {negativos} son negativas.\n"

    resp += "\nPrincipales comentarios:\n"
    for c in comentarios[:3]:
        resp += f"- {c}\n"

    return resp.strip()

# ============================
# 10. Generación híbrida
# ============================
def generate_answer(query: str, retrieved_docs: list[str]) -> str:
    """
    1. Intentamos con el modelo generativo.
    2. Si la salida parece rara / repetitiva / demasiado corta, usamos la redacción propia.
    """
    if not retrieved_docs:
        return "No tengo información suficiente para responder a esa pregunta."

    # primero intentamos que el modelo haga algo
    max_chars = 1200
    short_docs = [str(doc)[:max_chars] for doc in retrieved_docs]
    context = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(short_docs))

    prompt = (
        "Tienes opiniones de usuarios sobre vehículos (en español).\n"
        "Resume lo que dicen los usuarios sobre el tema consultado, de manera clara y corta.\n"
        "No repitas el texto exacto de las opiniones, sintetiza.\n"
        "Si no hay información suficiente, responde exactamente:\n"
        "\"No tengo información suficiente para responder a esa pregunta.\"\n\n"
        f"Opiniones:\n{context}\n\n"
        f"Pregunta: {query}\n"
        "Respuesta:"
    )

    model_text = ""
    try:
        out = generator(
            prompt,
            max_new_tokens=200,
            num_return_sequences=1,
            temperature=0.4,
            top_p=0.9,
            return_full_text=False,
        )
        model_text = out[0]["generated_text"]
        # limpieza mínima
        for token in ["Pregunta:", "Opiniones:", "Contexto:"]:
            if token in model_text:
                model_text = model_text.split(token)[0]
        model_text = " ".join(line.strip() for line in model_text.splitlines() if line.strip())
    except Exception:
        model_text = ""

    # criterio de “esto sirve”
    if model_text and len(model_text.split()) > 18 and not model_text.lower().startswith("comentario del usuario"):
        return model_text.strip()

    # si NO sirve lo del modelo, lo hacemos nosotros
    return build_human_answer(query, retrieved_docs)

# ============================
# 11. Pipeline
# ============================
def answer_pipeline(query: str) -> str:
    docs = retrieve_documents(query, k=3)

    # si pidió un año que no está, decimos qué años hay
    year = has_query_year(query)
    ctx = "\n\n".join(docs)
    if year and not context_has_year(ctx, year):
        yrs = extract_years_from_docs(docs)
        if yrs:
            return (
                f"No encuentro reseñas del {year} en los documentos recuperados.\n"
                f"Pero sí hay opiniones de estos años: {', '.join(yrs)}.\n"
                "¿Querés que te cuente sobre alguno de esos?"
            )
        else:
            return (
                f"No encuentro reseñas del {year} en los documentos recuperados. "
                "Solo hay comentarios de otros años o modelos."
            )

    return generate_answer(query, docs)
