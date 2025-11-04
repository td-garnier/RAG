import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# =====================================================
# 1️⃣  Chargement des variables d'environnement
# =====================================================
load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("⚠️ GOOGLE_API_KEY manquante dans le fichier .env")

# =====================================================
# 2️⃣  Chargement du modèle d'embeddings et de la base Chroma
# =====================================================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# =====================================================
# 3️⃣  Initialisation du modèle LLM Gemini
# =====================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3,
    max_tokens=700,
    timeout=None,
    max_retries=2
)

# =====================================================
# 4️⃣  Boucle interactive
# =====================================================
print("🤖 Assistant RAG prêt ! (tape 'exit' ou 'quit' pour arrêter)\n")

while True:
    query = input("🧠 Votre question : ").strip()
    if query.lower() in ["exit", "quit"]:
        print("\n👋 Fin de la session. À bientôt !")
        break

    # Recherche vectorielle
    results = vectordb.similarity_search(query, k=10)

    # Extraire les tags des résultats
    tags = [r.metadata.get("tag", "inconnu") for r in results]
    unique_tags = list(set(tags))

    print(f"\n🔍 {len(results)} extraits trouvés — tags détectés : {', '.join(unique_tags)}\n")

    # Concaténer les textes des documents trouvés
    if not results:
        print("❌ Aucun résultat pertinent trouvé dans la base.")
        continue

    rag_context = "\n\n".join(
        [f"Extrait {i+1} ({r.metadata.get('source', 'inconnu')}):\n{r.page_content}" for i, r in enumerate(results)]
    )

    # Préparer le prompt pour Gemini
    messages = [
        (
            "system",
            "Tu es un assistant spécialisé en documentation technique. "
            "Réponds uniquement à partir des extraits fournis ci-dessous, sans inventer d'informations."
        ),
        (
            "human",
            f"Voici les extraits trouvés par le RAG :\n{rag_context}\n\nQuestion : {query}"
        ),
    ]

    # Génération de la réponse
    try:
        ai_msg = llm.invoke(messages)
        print("\n🤖 Réponse :\n")
        print(ai_msg.content.strip(), "\n")

    except Exception as e:
        print(f"⚠️ Erreur lors de la génération de la réponse : {e}")
