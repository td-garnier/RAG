import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, AnyMessage


# =====================================================
# 1️⃣ Définition de l’état du graphe
# =====================================================
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str  

# =====================================================
# 2️⃣ Configuration de base
# =====================================================
load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("⚠️ GOOGLE_API_KEY manquante dans le fichier .env")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Mémoire conversationnelle persistante (Chroma)
history_db = Chroma(
    persist_directory="chroma_history",
    embedding_function=embeddings,
)

# Base documentaire RAG
vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
)

# Modèle LLM Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3,
    max_tokens=700,
    timeout=None,
    max_retries=2,
)


# =====================================================
# 3️⃣ Nœud principal : génération via le modèle
# =====================================================
def call_model(state: State):
    messages = state["messages"]
    user_query = state.get("user_query","")


    # 📜 Charger les 5 derniers échanges pour ce thread
    thread_history = []
    try:
        all_history = history_db.get(include=["metadatas", "documents"])
        for doc, meta in zip(all_history["documents"], all_history["metadatas"]):
            if meta.get("thread_id") == thread_id:
                content = getattr(doc, "page_content", doc)
                thread_history.append(content)
    except Exception as e:
        print(f"⚠️ Erreur lors de la récupération de l’historique : {e}")

    last_messages = thread_history[-5:]
    history_context = "\n".join(last_messages)

    # Construire les messages pour le LLM
    contextual_messages = [
        {
            "role": "system",
            "content": (
                "Tu es un assistant RAG. Réponds uniquement à partir des extraits fournis "
                "et de l’historique suivant :\n"
                f"{history_context}\n"
            ),
        }
    ]

    for m in messages:
        role = getattr(m, "role", getattr(m, "type", "user"))
        content = getattr(m, "content", "")
        contextual_messages.append({"role": role, "content": content})

    # Appel du modèle
    response = llm.invoke(contextual_messages)

    # Extraire le texte
    text_response = getattr(response, "content", "")
    if isinstance(text_response, list):
        text_response = " ".join([c.get("text", "") for c in text_response if isinstance(c, dict)])

    # Sauvegarder dans Chroma
    history_db.add_texts(
        texts=[f"USER: {user_query}\nAI: {text_response}"],
        metadatas=[{"thread_id": thread_id}],
    )

    return {"messages": [response]}




# =====================================================
# 4️⃣ Création du graphe LangGraph
# =====================================================
checkpointer = InMemorySaver()
builder = StateGraph(State)
builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)
graph = builder.compile(checkpointer=checkpointer)


# =====================================================
# 5️⃣ Boucle interactive
# =====================================================
thread_id = "main_thread"
print("🤖 Assistant RAG + Mémoire + LangGraph prêt !")
print("💬 Tapez votre question, ou 'history' pour voir les 5 derniers échanges.\n")

while True:
    query = input("🧠 Votre question : ").strip()

    # Sortie propre
    if query.lower() in ["exit", "quit"]:
        print("\n👋 Fin de la session. À bientôt !")
        break

    # Commande spéciale : afficher l’historique
    if query.lower() == "history":
        try:
            all_history = history_db.get(include=["metadatas", "documents"])
            thread_history = []
            for doc, meta in zip(all_history["documents"], all_history["metadatas"]):
                if meta.get("thread_id") == thread_id:
                    # doc peut être un str ou un Document
                    content = getattr(doc, "page_content", doc)
                    thread_history.append(content)

            if not thread_history:
                print("🕳️ Aucun historique trouvé pour ce thread.")
                continue

            print("\n📜 5 derniers échanges :\n")
            for entry in thread_history[-5:]:
                print(entry)
                print("-" * 40)

        except Exception as e:
            print(f"⚠️ Erreur lors de la récupération de l’historique : {e}")
        continue


    # Recherche vectorielle RAG
    results = vectordb.similarity_search(query, k=5)
    if not results:
        print("❌ Aucun résultat pertinent trouvé dans la base.")
        continue

    # Contexte documentaire
    rag_context = "\n\n".join(
        [f"📄 {r.metadata.get('source', 'inconnu')}:\n{r.page_content}" for r in results]
    )

    # Construire le prompt pour le modèle
    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un assistant RAG. Réponds uniquement à partir des extraits fournis "
                "et utilise l’historique conversationnel pour conserver le contexte."
            ),
        },
        {"role": "user", "content": f"Contexte documentaire :\n{rag_context}\n\nQuestion : {query}"},
    ]

    # Appel du graphe
    print("💭 Génération de la réponse...")
    try:
        output = graph.invoke(
            {"messages": messages,"user_query": query},
            config={"configurable": {"thread_id": thread_id}},
        )

        messages_out = output.get("messages", [])
        if not messages_out:
            print("⚠️ Aucune réponse générée.")
            continue

        last_msg = messages_out[-1]
        response_text = getattr(last_msg, "content", None)

        if not response_text:
            print(f"⚠️ Contenu vide : {last_msg}")
        else:
            print("\n🤖 Réponse :\n")
            print(response_text.strip(), "\n")

    except Exception as e:
        print(f"❌ Erreur pendant la génération : {e}")
