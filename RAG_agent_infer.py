import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

# LangChain/LangGraph/LLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, AnyMessage
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

# NOUVEAUX IMPORTS pour l'architecture Agent/Tool Calling
from langchain_core.tools import tool 
from langgraph.prebuilt import ToolNode, tools_condition 


# =====================================================
# 1️⃣ Définition de l’état du graphe (AgentState)
# =====================================================
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # 'rag_context' est conservé pour la clarté (bien qu'il soit maintenant géré par l'outil)
    rag_context: str
    history_context: str
    thread_id: str
    user_query: str


# =====================================================
# 2️⃣ Configuration de base (Inchangée)
# =====================================================
load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("⚠️ GOOGLE_API_KEY manquante dans le fichier .env")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

history_db = Chroma(
    persist_directory="chroma_history",
    embedding_function=embeddings,
)

vectordb = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3,
    max_tokens=700,
    timeout=None,
    max_retries=2,
)


# =====================================================
# 3️⃣ Définition des Outils et Nœuds
# =====================================================

# Définition de l'outil RAG (Inchangé)
@tool
def rag_search_tool(query: str) -> str:
    """Utilisez cet outil pour trouver des informations FACTUELLES dans la documentation technique (aspirateur, airfryer, etc.)."""
    
    results = vectordb.similarity_search(query, k=5)
    
    if not results:
        return "❌ Aucun résultat pertinent trouvé dans la base documentaire."
    
    rag_context = "\n\n".join(
        [f"📄 Source: {r.metadata.get('source', 'inconnu')}:\n{r.page_content}" for r in results]
    )
    return rag_context

# Initialisation de l'Agent LLM (Seul le LLM est lié aux outils)
tools = [rag_search_tool] 
agent_llm = llm.bind_tools(tools=tools) 


# Nœud 1 : Récupération de l'Historique (Inchangé)
def history_retriever_node(state: AgentState):
    """Récupère les 5 derniers échanges du thread actuel dans la base de mémoire (Chroma)."""
    thread_id = state["thread_id"]
    
    thread_history = []
    try:
        all_history = history_db.get(include=["metadatas", "documents"])
        for doc, meta in zip(all_history["documents"], all_history["metadatas"]):
            content = doc.get("page_content", doc) if isinstance(doc, dict) else str(doc)
            
            if meta.get("thread_id") == thread_id:
                thread_history.append(content)
    except Exception as e:
        print(f"⚠️ Erreur lors de la récupération de l’historique : {e}")
        
    history_context = "\n".join(thread_history[-5:])
    
    return {"history_context": history_context}


# Nœud 2 : Agent Node (Le Cerveau - Inchangé)
def agent_node(state: AgentState):
    """Reçoit l'historique des messages et décide d'appeler un outil ou de répondre."""
    messages = state["messages"]
    
    print("🧠 Agent: Décision...")
    
    # Le LLM, lié aux outils, décide s'il doit utiliser un outil ou répondre directement.
    response = agent_llm.invoke(messages)
    
    return {"messages": [response]} 


# NOUVEAU Nœud 3 : Sauvegarde (answer_node simplifié - Inchangé)
def answer_node(state: AgentState):
    """Gère la sauvegarde de l'échange final dans l'historique."""
    
    thread_id = state["thread_id"]
    user_query = state["user_query"]
    
    response_message = state["messages"][-1]
    text_response = getattr(response_message, "content", "")

    print(f"✅ Synthèse: Sauvegarde de l'échange.")
    
    try:
        history_db.add_texts(
            texts=[f"USER: {user_query}\nAI: {text_response}"],
            metadatas=[{"thread_id": thread_id}],
        )
    except Exception as e:
        print(f"⚠️ Erreur lors de la sauvegarde de l’historique : {e}")

    return {"messages": [response_message]} 


# =====================================================
# 4️⃣ Création du Graphe LangGraph (Utilisation de ToolNode)
# =====================================================
checkpointer = InMemorySaver()
builder = StateGraph(AgentState)

# Ajouter les nœuds
builder.add_node("history_retriever_node", history_retriever_node)
builder.add_node("agent_node", agent_node)
builder.add_node("tool_executor", ToolNode(tools)) 
builder.add_node("answer_node", answer_node)

# 1. Début -> Récupération de l'historique
builder.add_edge(START, "history_retriever_node")

# 2. Après l'historique, l'Agent prend la première décision
builder.add_edge("history_retriever_node", "agent_node")

# 3. Boucle conditionnelle Agent ↔ Outil
builder.add_conditional_edges(
    "agent_node",
    # tools_condition est une fonction utilitaire qui vérifie si l'Agent a demandé un outil (tool_calls)
    tools_condition,
    {
        # Si 'tools' est retourné (Agent demande un outil) -> Exécuter l'outil
        "tools": "tool_executor", 
        # Si 'END' est retourné (Agent a généré la réponse finale) -> Fin/Sauvegarde
        "end": "answer_node",   
    },
)

# Après l'exécution de l'outil, on retourne à l'Agent pour qu'il analyse le résultat et réponde
builder.add_edge("tool_executor", "agent_node")

# Fin
builder.add_edge("answer_node", END)

# Compilation du graphe
agent_tool_calling_graph = builder.compile(checkpointer=checkpointer)


# =====================================================
# 5️⃣ Boucle interactive de l'Agent (Inchangée)
# =====================================================
thread_id = "main_thread"
print("🤖 Agent Tool Calling + RAG + Mémoire + LangGraph prêt !")
print("💬 Tapez votre question, ou 'history' pour voir les 5 derniers échanges.\n")

while True:
    query = input("🧠 Votre question : ").strip()

    if query.lower() in ["exit", "quit"]:
        print("\n👋 Fin de la session. À bientôt !")
        break
    
    if query.lower() == "history":
        try:
            all_history = history_db.get(include=["metadatas", "documents"])
            thread_history = []
            for doc, meta in zip(all_history["documents"], all_history["metadatas"]):
                content = doc.get("page_content", doc) if isinstance(doc, dict) else str(doc)
                if meta.get("thread_id") == thread_id:
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


    initial_state = {
        "messages": [HumanMessage(content=query)], 
        "thread_id": thread_id,
        "user_query": query,
        "rag_context": "", 
        "history_context": "" 
    }

    print("💭 L'Agent réfléchit (Boucle Tool Calling)...")
    
    checkpointer_config = {
        "configurable": {
            "thread_id": thread_id 
        }
    }
    
    try:
        output = agent_tool_calling_graph.invoke(initial_state, config=checkpointer_config)

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