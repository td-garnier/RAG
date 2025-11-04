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


# =====================================================
# 2️⃣ Configuration de base
# =====================================================
load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("⚠️ GOOGLE_API_KEY manquante dans le fichier .env")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3,
    max_tokens=700,
    timeout=None,
    max_retries=2
)


# =====================================================
# 3️⃣ Définition du nœud de génération
# =====================================================
def call_model(state: State):
    """Nœud : appelle Gemini avec l’historique et renvoie la réponse."""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# =====================================================
# 4️⃣ Création et compilation du graphe LangGraph
# =====================================================
checkpointer = InMemorySaver()
builder = StateGraph(State)

# Ajoute ton nœud principal
builder.add_node("model", call_model)

# 🧠 définir le flux
builder.add_edge(START, "model")  # entrée → model
builder.add_edge("model", END)    # model → fin

# Compilation
graph = builder.compile(checkpointer=checkpointer)


# =====================================================
# 5️⃣ Boucle interactive avec mémoire persistante
# =====================================================
thread_id = "main_thread"
print("🤖 Assistant RAG + LangGraph prêt ! (tape 'exit' ou 'quit' pour arrêter)\n")

while True:
    query = input("🧠 Votre question : ").strip()
    if query.lower() in ["exit", "quit"]:
        print("\n👋 Fin de la session. À bientôt !")
        break

    # Recherche vectorielle RAG
    results = vectordb.similarity_search(query, k=5)
    if not results:
        print("❌ Aucun résultat pertinent trouvé dans la base.")
        continue

    # Construire le contexte à partir des extraits trouvés
    rag_context = "\n\n".join(
        [f"📄 {r.metadata.get('source', 'inconnu')}:\n{r.page_content}" for r in results]
    )

    # Construire le prompt pour le LLM
    messages = [
        {"role": "system", "content": "Tu es un assistant RAG. Réponds uniquement à partir des extraits fournis."},
        {"role": "user", "content": f"Contexte :\n{rag_context}\n\nQuestion : {query}"}
    ]

    # Appel du graphe avec historique LangGraph
    print("💭 Génération de la réponse...")
    try:
        output = graph.invoke(
            {"messages": messages},
            config={"configurable": {"thread_id": thread_id}},
        )

        # Vérification de la structure de la sortie
        messages_out = output.get("messages", [])
        if not messages_out:
            print("⚠️ Aucune réponse générée par le modèle.")
            continue

        last_msg = messages_out[-1]
        response_text = getattr(last_msg, "content", None)

        if not response_text:
            print(f"⚠️ Contenu vide. Message brut : {last_msg}")
        else:
            print("\n🤖 Réponse :\n")
            print(response_text.strip(), "\n")

    except Exception as e:
        print(f"❌ Erreur pendant la génération : {e}")