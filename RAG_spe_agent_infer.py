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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field


# =====================================================
# 1️⃣ Définition de l’état du graphe (AgentState)
# =====================================================
class AgentState(TypedDict):
    # 'messages' est géré par LangGraph (add_messages)
    messages: Annotated[list[AnyMessage], add_messages]
    # 'rag_context' stocke le résultat de la recherche vectorielle (si effectuée)
    rag_context: str
    # 'history_context' stocke l'historique de conversation récupéré de Chroma
    history_context: str
    # 'thread_id' est explicitement ajouté à l'état pour les nœuds
    thread_id: str
    # 'user_query' est conservé pour la sauvegarde finale
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
# 3️⃣ Définition des Nœuds de l'Agent
# =====================================================

# Nœud 1 : Récupération de l'Historique (Toujours)
def history_retriever_node(state: AgentState):
    """Récupère les 5 derniers échanges du thread actuel dans la base de mémoire (Chroma)."""
    thread_id = state["thread_id"]
    
    thread_history = []
    try:
        all_history = history_db.get(include=["metadatas", "documents"])
        for doc, meta in zip(all_history["documents"], all_history["metadatas"]):
            # Utilisation de .get("page_content") pour gérer les différents types de documents Chroma
            content = doc.get("page_content", doc) if isinstance(doc, dict) else str(doc)
            
            if meta.get("thread_id") == thread_id:
                thread_history.append(content)
    except Exception as e:
        print(f"⚠️ Erreur lors de la récupération de l’historique : {e}")
        
    history_context = "\n".join(thread_history[-5:])
    
    # Stocker l'historique dans l'état pour les nœuds suivants
    return {"history_context": history_context}


# 🆕 Nœud 2 : Router (Décision LLM)
class RouteDecision(BaseModel):
    action: str = Field(description="Must be 'use_rag' if an external search is required, or 'final_answer' otherwise.")

parser = PydanticOutputParser(pydantic_object=RouteDecision)

router_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "Tu es un routeur intelligent. Analyse la question de l'utilisateur en tenant compte de l'Historique. "
     "Si la réponse est déjà contenue dans l'Historique, choisis 'final_answer'. "
     "Si la question est nouvelle, factuelle, et nécessite une recherche documentaire, choisis 'use_rag'. "
     "Sinon, si elle est purement conversationnelle (salutations, transition), choisis 'final_answer'. "
     "Réponds uniquement avec un objet JSON qui correspond au schéma donné."
     "\n\n--- Historique de conversation ---\n{history}"), # ⬅️ Ajout de l'historique
     
    ("human", "Question à analyser: {question}"), 
])

router_chain = router_prompt | llm.with_structured_output(RouteDecision) | (lambda x: x.action)

def router_node(state: AgentState) -> str:
    """Nœud de routage conditionnel."""
    last_message = state["messages"][-1].content 
    history_context = state["history_context"] # ⬅️ Récupération de l'historique
    
    print(f"🔄 Routeur: Analyse de la question...")
    
    # Appel de la chaîne de routage avec l'historique
    decision = router_chain.invoke({
        "question": last_message,
        "history": history_context # ⬅️ Passage de la variable 'history'
    })
    
    if decision == "use_rag":
        print("🔍 Routeur: Décision = Recherche RAG")
        return "rag_search_node"
    
    print("💬 Routeur: Décision = Réponse directe (Historique suffisant)")
    return "answer_node"


# 🆕 Nœud 3 : Recherche RAG (Exécution d'outil conditionnelle)
def rag_search_node(state: AgentState):
    """Exécute la recherche vectorielle RAG et met à jour l'état avec le contexte trouvé."""
    last_message = state["messages"][-1].content
    
    results = vectordb.similarity_search(last_message, k=5)
    
    if not results:
        print("❌ RAG: Aucun résultat pertinent trouvé.")
        rag_context = "❌ Aucun résultat pertinent trouvé dans la base documentaire."
    else:
        print(f"✅ RAG: {len(results)} documents trouvés.")
        rag_context = "\n\n".join(
            [f"📄 Source: {r.metadata.get('source', 'inconnu')}:\n{r.page_content}" for r in results]
        )
    
    # Stocker le contexte RAG dans l'état
    return {"rag_context": rag_context}


# 🆕 Nœud 4 : Synthèse et Sauvegarde (Réponse Finale)
def answer_node(state: AgentState):
    """Synthétise la réponse en utilisant le contexte RAG, l'historique, et sauvegarde l'échange."""
    
    # 1. Préparation des contextes
    thread_id = state["thread_id"]
    user_query = state["user_query"]
    messages = state["messages"]
    history_context = state["history_context"] 
    # rag_context sera vide si rag_search_node n'a pas été appelé
    rag_context = state.get("rag_context", "Aucun contexte documentaire supplémentaire n'a été jugé nécessaire.")
    
    # 2. Construction du Prompt Système (Amélioré pour gérer le cas RAG vide)
    system_prompt = (
        "Tu es un assistant RAG. Réponds de manière précise et professionnelle. "
        "Si le Contexte RAG contient des informations pertinentes, utilise-les comme source principale. "
        "Sinon, utilise l'Historique de conversation pour conserver le contexte. "
        "Ne fais référence au 'Contexte RAG' que si tu l'utilises."
        "\n\n--- Contexte RAG ---\n"
        f"{rag_context}"
        "\n\n--- Historique de conversation (5 derniers échanges) ---\n"
        f"{history_context}"
    )

    # 3. Appel du modèle
    system_message_obj = SystemMessage(content=system_prompt)

    # Le tableau final doit contenir uniquement des objets BaseMessage
    final_messages = [system_message_obj]

    # Ajouter le dernier message utilisateur (qui est déjà un objet BaseMessage)
    final_messages.append(state["messages"][-1]) 

    response = llm.invoke(final_messages)
    text_response = getattr(response, "content", "")

    # 4. Sauvegarde dans Chroma (Mémoire conversationnelle)
    try:
        history_db.add_texts(
            texts=[f"USER: {user_query}\nAI: {text_response}"],
            metadatas=[{"thread_id": thread_id}],
        )
    except Exception as e:
        print(f"⚠️ Erreur lors de la sauvegarde de l’historique : {e}")

    # 5. Retourner la réponse
    return {"messages": [response]}


# =====================================================
# 4️⃣ Création du Graphe LangGraph (Orchestration)
# =====================================================
checkpointer = InMemorySaver()
builder = StateGraph(AgentState)

# Ajouter les nœuds
builder.add_node("history_retriever_node", history_retriever_node)
builder.add_node("rag_search_node", rag_search_node)
builder.add_node("answer_node", answer_node)

# Définir les chemins (Edges)
builder.add_edge(START, "history_retriever_node")
# Le nœud history_retriever_node passe le relais à la fonction router_node
builder.add_conditional_edges(
    "history_retriever_node", # ⬅️ Démarre le routage après le 'history_retriever_node'
    router_node,              # ⬅️ Utilise la fonction router_node pour la décision
    {
        "rag_search_node": "rag_search_node", # Si besoin de RAG -> Recherche
        "answer_node": "answer_node",         # Si pas besoin de RAG -> Réponse directe
    },
)

# Après la recherche RAG, on va toujours à la réponse
builder.add_edge("rag_search_node", "answer_node")

# Fin
builder.add_edge("answer_node", END)

# Compilation du graphe
agent_rag_graph = builder.compile(checkpointer=checkpointer)


# =====================================================
# 5️⃣ Boucle interactive de l'Agent
# =====================================================
thread_id = "main_thread"
print("🤖 Agent RAG + Routage + Mémoire + LangGraph prêt !")
print("💬 Tapez votre question, ou 'history' pour voir les 5 derniers échanges.\n")

while True:
    query = input("🧠 Votre question : ").strip()

    # Sortie propre et commande 'history' (inchangées)
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


    # 🆕 Préparation de l'entrée pour le Graphe Agent
    # Le graphe reçoit la question utilisateur, l'ID du thread et la query brute.
    initial_state = {
        "messages": [HumanMessage(content=query)], 
        "thread_id": thread_id,
        "user_query": query,
        "rag_context": "", # Initialisé vide
        "history_context": "" # Initialisé vide
    }

    # Appel du graphe
    print("💭 L'Agent réfléchit (Routage et Génération)...")
    # 1. ⚠️ Définir la configuration du Checkpointer
    checkpointer_config = {
        "configurable": {
            "thread_id": thread_id # Indique au Checkpointer où sauvegarder/charger
        }
    }
    try:
        # Lancement du graphe avec l'état initial ET la configuration du checkpointer.
        output = agent_rag_graph.invoke(initial_state,config=checkpointer_config)

        # Extraction et affichage de la réponse finale
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