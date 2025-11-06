#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # ============================
# rag_backend.py (versión para Streamlit Cloud)
# ============================
import os
import re
import warnings
import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer
from transformers import pipeline

warnings.filterwarnings("ignore")

# =========================================================
# RUTAS RELATIVAS (IMPORTANTE PARA STREAMLIT CLOUD)
# =========================================================
DATA_PATH = "Car_Reviews_español.xlsx"
EMB_PATH = "car_reviews_embeddings.npy"

# =========================================================
# 1. CARGA DE DATOS
# =========================================================
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

print(f"✅ Dataset cargado: {len(df)} reseñas")

# =========================================================
# 2. MODELO DE EMBEDDINGS
# =========================================================
print("Cargando modelo de SentenceTransformer - all-MiniLM-L6-v2 ...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
print("Modelo cargado ✅")

# =========================================================
# 3. EMBEDDINGS (cargar o generar)
# =========================================================
if os.path.exists(EMB_PATH):
    print(f"Cargando embeddings desde disco: {EMB_PATH}")
    doc_embeddings = np.load(EMB_PATH)
else:
    print("Generando embeddings para el corpus (solo la primera vez)...")
    doc_embeddings = embedding_model.encode(documents, show_progress_bar=False)
    doc_embeddings = np.array(doc_embeddings, dtype="float32")
    # en cloud a veces no se puede escribir, así que lo envolvemos
    try:
        np.save(EMB_PATH, doc_embeddings)
    except Exception:
        pass
    print("✅ embeddings generados")

doc_embeddings = np.array(doc_embeddings, dtype="float32")
print(f"Embeddings listos. Forma: {doc_embeddings.shape}")

# =========================================================
# 4. FAISS
# =========================================================
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)
print(f"Índice FAISS creado con {index.ntotal} vectores.")

# =========================================================
# 5. MODELO GENERATIVO
# =========================================================
print("Cargando modelo generativo (GPT-2 español)...")
generator = pipeline(
    task="text-generation",
    model="datificate/gpt2-small-spanish",
    device=-1
)
print("Modelo generativo cargado ✅")

# =========================================================
# 6. LISTAS / EXPANSIONES (igual que el tuyo)
# =========================================================
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

# =========================================================
# 7. HELPERS (igual que el tuyo)
# =========================================================
def expand_query(query: str) -> str:
    q = query.lower()
    extra = []
    for key, vals in QUERY_EXPANSIONS.items():
        if key in q:
            extra.extend(vals)
    if extra:
        return query + " " + " ".join(extra)
    return query

def has_query_year(query: str) -> str | None:
    m = re.search(r"\b(19|20)\d{2}\b", query)
    if m:
        return m.group(0)
    return None

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
            s_low = sent.lower()
            if any(term in s_low for term in aspect_terms):
                filtered_chunks.append(sent.strip() + ".")
    return filtered_chunks or retrieved_docs

def extract_models_from_docs(docs: list[str]) -> list[str]:
    models = []
    for d in docs:
        if d.startswith("Vehículo:"):
            m = d.split("Comentario del usuario:")[0].strip()
            m = m.replace("Vehículo:", "").strip().strip(".")
            models.append(m)
    return models

def extract_years_from_docs(docs: list[str]) -> list[str]:
    years = set()
    for d in docs:
        found_full = re.findall(r"\b(?:19|20)\d{2}\b", d)
        for y in found_full:
            years.add(y)
    return sorted(list(years))

def extractive_answer(query: str, retrieved_docs: list) -> str:
    aspect_docs = filter_docs_by_aspect(retrieved_docs, query)
    if not aspect_docs:
        return "No tengo información suficiente para responder a esa pregunta."
    bullets = []
    for i, doc in enumerate(aspect_docs, start=1):
        text = str(doc).strip()
        bullets.append(f"- {text} [{i}]")
    return "Lo que dicen los usuarios es:\n" + "\n".join(bullets)

# =========================================================
# 8. RECUPERACIÓN (igual que el tuyo)
# =========================================================
def retrieve_documents(query: str, k: int = 2):
    q_low = query.lower()
    brands_in_query = [b for b in KNOWN_BRANDS if b in q_low]
    models_in_query = [m for m in KNOWN_MODELS if m in q_low]

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

    expanded_query = expand_query(query)
    query_embedding = embedding_model.encode([expanded_query])
    query_embedding = np.array(query_embedding, dtype="float32")
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

# =========================================================
# 9. GENERACIÓN (con seguro, igual que te dije)
# =========================================================
def generate_answer(query: str, retrieved_docs: list):
    if not retrieved_docs:
        return "No tengo información suficiente para responder a esa pregunta."

    max_chars = 1200
    retrieved_docs = [str(doc)[:max_chars] for doc in retrieved_docs]
    context = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(retrieved_docs))

    prompt = (
        "Tienes opiniones de usuarios sobre vehículos (en español).\n"
        "Tu tarea es contestar a la pregunta RESUMIENDO lo que dicen los usuarios.\n"
        "No inventes modelos ni años que no aparezcan en las opiniones.\n"
        "Si no hay información suficiente, responde exactamente:\n"
        "\"No tengo información suficiente para responder a esa pregunta.\"\n\n"
        f"Opiniones:\n{context}\n\n"
        f"Pregunta: {query}\n"
        "Resumen de lo que dicen los usuarios:"
    )

    generated = generator(
        prompt,
        max_new_tokens=200,
        num_return_sequences=1,
        temperature=0.3,
        top_p=0.95,
        top_k=50,
        return_full_text=False
    )

    raw_answer = generated[0]["generated_text"]

    for token in ["Pregunta:", "Opiniones:", "Contexto:"]:
        if token in raw_answer:
            raw_answer = raw_answer.split(token)[0]

    cleaned = " ".join(line.strip() for line in raw_answer.splitlines() if line.strip())

    bad_starts = [
        "no inventes modelos",
        "no inventes",
        "tienes opiniones de usuarios",
    ]
    if (not cleaned) or len(cleaned.split()) < 5 or any(cleaned.lower().startswith(b) for b in bad_starts):
        return extractive_answer(query, retrieved_docs)

    return cleaned

# =========================================================
# 10. PIPELINE (misma lógica que el tuyo)
# =========================================================
def answer_pipeline(query: str) -> str:
    docs = retrieve_documents(query, k=3)

    # si pidieron un año que no está
    year = has_query_year(query)
    context_text = "\n\n".join(docs)
    if year and not context_has_year(context_text, year):
        available_years = extract_years_from_docs(docs)
        if available_years:
            return (
                f"No encuentro reseñas del {year} en los documentos recuperados.\n"
                f"Pero sí tengo reseñas de estos años: {', '.join(available_years)}.\n"
                "¿Sobre cuál de esos años querés saber más?"
            )
        else:
            return (
                f"No encuentro reseñas del {year} en los documentos recuperados. "
                "Solo hay comentarios de otros años/modelos."
            )

    # si la pregunta es de un aspecto y recuperamos distintos modelos, en Jupyter repreguntábamos
    # en la web lo mostramos como está
    aspect_docs = filter_docs_by_aspect(docs, query)
    return generate_answer(query, aspect_docs)


# In[ ]:




