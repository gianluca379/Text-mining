#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import importlib

st.set_page_config(page_title="RAG de reseñas de autos", layout="wide")

st.title("🚗 Asistente RAG sobre reseñas de vehículos")
st.write(
    "Escribí una pregunta sobre las reseñas del Excel (consumo, ruido, transmisión, etc.). "
    "La app intentará cargar el motor de RAG."
)

@st.cache_resource
def load_backend():
    """
    Intentamos importar el backend (rag_backend.py).
    Si falla por torch / dll / numpy, devolvemos la excepción
    en lugar de romper Streamlit.
    """
    try:
        backend = importlib.import_module("rag_backend")
        return backend
    except Exception as e:
        return e

backend = load_backend()

# UI principal
query = st.text_area("Pregunta:", value="¿Qué dicen sobre el consumo de la Ford Transit?", height=100)
k_docs = st.slider("Cantidad de documentos a mostrar", 1, 5, 2, 1)

if st.button("Consultar"):
    if isinstance(backend, Exception):
        st.error(
            "No pude cargar el backend de RAG.\n\n"
            f"Detalle técnico: {type(backend).__name__}: {backend}\n\n"
            "Esto suele pasar cuando Streamlit se ejecuta en un entorno distinto al de Jupyter "
            "(por ejemplo, falta PyTorch o sentence-transformers). "
            "Volvé a correr `streamlit run app.py` desde el mismo entorno donde te anduvo el RAG."
        )
    else:
        if query.strip():
            with st.spinner("Buscando y generando respuesta..."):
                # usamos las funciones que ya tenés en tu archivo bueno
                answer = backend.answer_pipeline(query)

                st.subheader("🟢 Respuesta")
                st.write(answer)

                st.subheader("📄 Documentos recuperados")
                docs = backend.retrieve_documents(query, k=k_docs)
                for i, d in enumerate(docs, start=1):
                    with st.expander(f"Documento {i}"):
                        st.write(d)
        else:
            st.warning("Escribí primero una pregunta.")
else:
    if isinstance(backend, Exception):
        st.warning(
            "⚠️ El motor RAG todavía no se pudo cargar. "
            "Cuando lo ejecutes desde el entorno correcto, podés apretar 'Consultar' y va a funcionar."
        )


