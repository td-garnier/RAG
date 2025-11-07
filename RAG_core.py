# RAG_core.py

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document

load_dotenv()

# --- Initialisation des bases de données (Core) ---
# Ces variables sont globales et initialisées une seule fois à l'importation.
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("⚠️ GOOGLE_API_KEY manquante dans le fichier .env")
    
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Les bases Chroma sont nécessaires pour les outils RAG et LTM
vectordb_rag_global = Chroma(embedding_function=embeddings, persist_directory="chroma_db")
vectordb_history_global = Chroma(embedding_function=embeddings, persist_directory="chroma_history")


def retrieve_context_core(query: str, k=4) -> str:
    """
    Fonction de récupération de contexte pure RAG.
    Récupère les documents et les formate pour le LLM SANS dépendance Chainlit.
    """
    
    # 1. Utilisez le retriever sur la base RAG
    retriever_rag = vectordb_rag_global.as_retriever(search_kwargs={"k": k})
    source_documents: List[Document] = retriever_rag.invoke(query)
    
    # 2. Formatage du contexte
    context = ""
    source_names_list = []

    for i, doc in enumerate(source_documents):
        source = doc.metadata.get('source', 'Inconnu')
        page = doc.metadata.get('page_label', 1)
        citation_name = f"source_{i+1}"
        content = doc.page_content

        source_names_list.append(citation_name)
        
        # Construction du contexte RAG pour l'Agent/LLM
        context += f"[DOCUMENT RAG {i+1} - CITATION: {citation_name} (Source: {source}, Page: {page})]: {content}\n---\n"
        
    # Ajout d'une instruction forte pour forcer la citation des sources
    citation_instruction = f"\n\n**INSTRUCTION LLM:** Réponds en utilisant UNIQUEMENT les sources ci-dessus et ajoute OBLIGATOIREMENT à la fin de ta réponse la liste des sources citées sous la forme : **Sources: {', '.join(source_names_list)}**."

    return context + citation_instruction

def get_rag_documents(query: str, k: int = 4) -> List[Document]:
    """Récupère les objets Document bruts à partir de Chroma pour l'évaluation."""
    # Assurez-vous que vectordb_rag_global est correctement initialisé avant cet appel.
    retriever_rag = vectordb_rag_global.as_retriever(search_kwargs={"k": k})
    return retriever_rag.invoke(query)