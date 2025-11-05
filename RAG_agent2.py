import os
import sys 
from dotenv import load_dotenv

# 🆕 --- IMPORTS PHOENIX & INSTRUMENTATION ---
import phoenix as ph
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor 
from opentelemetry import trace
from phoenix.evals import (
    HallucinationEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
    OpenAIModel,
    run_evals
)
# --- FIN IMPORTS PHOENIX ---

# LangChain/LLM/RAG/Agents
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent 
from langchain_core.tools import tool 
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# LangGraph (pour le Checkpoint/Mémoire uniquement)
from langgraph.checkpoint.memory import InMemorySaver 

# --- VARIABLES GLOBALES ---
thread_id = "user_123_session"
# 🆕 NOM DU PROJET POUR ORGANISER LES TRACES DANS PHOENIX
PROJECT_NAME = "RAG_Agent_LTM_Tracing" 
session = None
vectordb_rag = None
vectordb_history = None
llm = None
agent = None

# =====================================================
# 0️⃣ INITIALISATION PHOENIX (Tracing)
# =====================================================
print(f"🚀 Démarrage de Phoenix pour le projet : **{PROJECT_NAME}**...")
try:
    # 1. Lance l'application Phoenix et spécifie le nom du projet
    session = ph.launch_app()
    print(f"📈 Phoenix UI démarré ! Consultez : {session.url}")
    
    # configure the Phoenix tracer
    tracer_provider = register(
        project_name=PROJECT_NAME, # Default is 'default'
        endpoint="http://localhost:6006/v1/traces"
        )

    # 2. Instrumente LangChain pour l'envoi automatique des traces
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    print("✅ Instrumentation LangChain/Phoenix réussie.")

    tracer = tracer_provider.get_tracer(__name__)

except Exception as e:
    print(f"❌ Erreur critique lors du démarrage/instrumentation de Phoenix. Le tracing sera désactivé: {e}")
    session = None

print("------------------------------------------------------------------")

# =====================================================
# 1️⃣ Configuration de base et Modèle (Inchangé)
# =====================================================
load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("⚠️ GOOGLE_API_KEY manquante dans le fichier .env")

# Configuration des Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 1. Base Chroma pour les Documents (RAG)
vectordb_rag = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db",
)

# 2. Base Chroma pour l'Historique (Mémoire à Long Terme / LTM)
vectordb_history = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_history",
)

# Modèle LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    temperature=0.3,
)

# Configuration du Checkpointer LangGraph
checkpointer = InMemorySaver() 

# =====================================================
# 2️⃣ Définition des Outils (Tool) (Inchangé)
# =====================================================

@tool
def retrieve_context(query: str) -> str:
    """
    Récupère les documents les plus pertinents de la base de données 
    documentaire (RAG) pour répondre à une question factuelle.
    """
    results = vectordb_rag.similarity_search(query, k=3) 
    if not results:
        return "❌ Aucun document RAG pertinent trouvé."
    
    rag_context = "\n\n".join(
        [f"📄 Source: {r.metadata.get('source', 'inconnu')}:\n{r.page_content[:200]}..." for r in results]
    )
    return rag_context

@tool
def retrieve_history(query: str) -> str:
    """
    Récupère des fragments de conversations passées pertinentes de la mémoire 
    historique (chroma_history) pour le contexte à long terme.
    """
    results = vectordb_history.similarity_search(query, k=2) 
    
    if not results:
        return "❌ Aucun fragment de conversation passée pertinent trouvé."
    
    history_context = "\n\n".join(
        [f"💬 Fragment de conversation: {r.page_content[:200]}..." for r in results]
    )
    return history_context

tools = [retrieve_context, retrieve_history]

# =====================================================
# 3️⃣ Création et Utilitaires de l'Agent (Inchangé)
# =====================================================

system_prompt_text = (
    "Tu es un assistant RAG professionnel avec une mémoire à long terme. "
    "Tu as accès à deux outils : 'retrieve_context' pour les faits documentaires et "
    "répondre à une question factuelle et 'retrieve_history' pour te rappeler des discussions antérieures."
    "Utilise l'outil le plus approprié pour chaque requête. Réponds poliment."
)

agent = create_agent(
    llm, 
    tools, 
    system_prompt=system_prompt_text, 
    checkpointer=checkpointer, 
)

def save_to_long_term_memory(thread_id: str, user_query: str, ai_response: str):
    """Enregistre la paire de messages dans la base Chroma pour la LTM."""
    global vectordb_history
    content = f"Utilisateur ({thread_id}): {user_query}\nIA ({thread_id}): {ai_response}"
    vectordb_history.add_texts([content], metadata={"thread_id": thread_id})


# =====================================================
# 4️⃣ Boucle interactive de l'Agent avec Streaming (Inchangé)
# =====================================================
print(f"🤖 Agent LangChain (create_agent) + Streaming + Mémoire LTM/STM prêt ! (ID : {thread_id})")
print(f"✅ Outils RAG/Historique. La réponse sera affichée en temps réel.\n")

while True:
    query = input("🧠 Votre question : ").strip()

    if query.lower() in ["exit", "quit"]:
        print("\n👋 Fin de la session. À bientôt !")
        break

    initial_messages = [HumanMessage(content=query)]
    checkpointer_config = {
        "configurable": {
            "thread_id": thread_id 
        }
    }

    print("💭 L'Agent réfléchit (Streaming)...")
    print("\n🤖 Réponse :\n")
    final_response_text = ""
    
    try:
        for token, metadata in agent.stream(
            {"messages": initial_messages},
            config=checkpointer_config,
            stream_mode="messages",
        ):
            message = token
            
            if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                print(message.content, end="", flush=True)
                final_response_text += message.content
            
        print("\n")

        save_to_long_term_memory(thread_id, query, final_response_text)
        print("💾 Le tour de conversation a été enregistré dans 'chroma_history'.\n")

    except Exception as e:
        print(f"\n❌ Erreur pendant la génération : {e}")