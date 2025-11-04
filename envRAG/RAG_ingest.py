import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI

# =====================================================
# Configuration
# =====================================================
pdf_folder = "./documents"
persist_dir = "chroma_db"

load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("⚠️ GOOGLE_API_KEY manquante dans le fichier .env")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# =====================================================
# Charger la base Chroma existante (si elle existe)
# =====================================================
vectordb = None
if os.path.exists(persist_dir):
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    print("📦 Base Chroma existante chargée.")
else:
    print("🆕 Aucune base Chroma trouvée — une nouvelle sera créée.")

# =====================================================
# Identifier les fichiers déjà indexés
# =====================================================
indexed_files = set()
if vectordb is not None:
    try:
        collection = vectordb._collection.get(include=["metadatas"])
        if collection and "metadatas" in collection:
            for meta in collection["metadatas"]:
                if meta and "source" in meta:
                    indexed_files.add(meta["source"])
        print(f"🔍 {len(indexed_files)} fichiers déjà indexés : {indexed_files}")
    except Exception as e:
        print(f"⚠️ Impossible de lire les métadonnées : {e}")

# =====================================================
# Préparation du LLM pour la génération de tags
# =====================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    max_tokens=200,
    timeout=None,
    max_retries=2
)

# =====================================================
# Parcours des nouveaux PDF uniquement
# =====================================================
all_docs = []
for file_name in os.listdir(pdf_folder):
    if not file_name.lower().endswith(".pdf"):
        continue
    if file_name in indexed_files:
        print(f"⏭️ Fichier déjà indexé : {file_name}")
        continue

    file_path = os.path.join(pdf_folder, file_name)
    print(f"📄 Nouveau fichier détecté : {file_name}")

    try:
        # Charger le texte uniquement via PyPDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        content_excerpt = " ".join([d.page_content[:2000] for d in docs])

        # Génération des tags
        messages = [
            ("system", "Tu lis un document et génères 2 ou 3 mots-clés résumant le sujet principal."),
            ("human", f"Voici le texte : {content_excerpt}\nDonne seulement un ou deux mots-clés séparés par des virgules.")
        ]
        ai_msg = llm.invoke(messages)
        tag = ai_msg.content.strip()
        print(f"🧩 Tags générés pour {file_name} : {tag}")

        # Ajouter les métadonnées
        for doc in docs:
            doc.metadata["source"] = file_name
            doc.metadata["tag"] = tag

        all_docs.extend(docs)

    except Exception as e:
        print(f"⚠️ Erreur lors du chargement de {file_name}: {e}")

# =====================================================
# Si aucun nouveau document, on arrête
# =====================================================
if not all_docs:
    print("✅ Aucun nouveau document à ajouter.")
    exit()

# =====================================================
# Split et ajout à la base
# =====================================================
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = splitter.split_documents(all_docs)

print(f"🧩 {len(all_splits)} nouveaux chunks à indexer.")

if vectordb is None:
    vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory=persist_dir)
    print("✅ Nouvelle base vectorielle créée.")
else:
    vectordb.add_documents(all_splits)
    print("✅ Nouveaux documents ajoutés à la base existante.")

vectordb.persist()
print("💾 Base vectorielle sauvegardée avec succès.")
