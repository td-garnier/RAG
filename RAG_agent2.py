import os
import sys 
from dotenv import load_dotenv

# 🆕 --- IMPORTS CHAINLIT ---
import chainlit as cl
# --- FIN IMPORTS CHAINLIT ---

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
    run_evals,
)
import pandas as pd 
# --- FIN IMPORTS PHOENIX ---

# LangChain/LLM/RAG/Agents
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent 
from langchain_core.tools import tool 
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# LangGraph (pour le Checkpoint/Mémoire uniquement)
from langgraph.checkpoint.memory import InMemorySaver 

# --- VARIABLES GLOBALES ---
thread_id = "user_123_session"
PROJECT_NAME = "RAG_Agent_LTM_Tracing" 
# Les variables globales LLM/Agent sont maintenant initialisées dans cl.on_chat_start
# Nous conservons la session Phoenix au niveau global pour l'évaluation de fin
global session 
session = None

# =====================================================
# 0️⃣ INITIALISATION PHOENIX (Tracing)
# =====================================================

# Nous laissons l'initialisation Phoenix se faire au démarrage du script, 
# car elle doit se faire avant que l'instrumentation LangChain ne soit utilisée.
print(f"🚀 Démarrage de Phoenix pour le projet : **{PROJECT_NAME}**...")
try:
    # 1. Lance l'application Phoenix
    session = ph.launch_app()
    print(f"📈 Phoenix UI démarré ! Consultez : {session.url}")
    
    # 2. Configure le Phoenix tracer (votre configuration validée)
    tracer_provider = register(
        project_name=PROJECT_NAME,
        endpoint="http://localhost:6006/v1/traces"
    )

    # 3. Instrumente LangChain avec le tracer provider spécifique
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    print("✅ Instrumentation LangChain/Phoenix réussie.")

    # 4. Récupère le tracer (non strictement nécessaire ici, car l'agent est auto-instrumenté)
    tracer = tracer_provider.get_tracer(__name__)

except Exception as e:
    print(f"❌ Erreur critique lors du démarrage/instrumentation de Phoenix. Le tracing sera désactivé: {e}")
    session = None

print("------------------------------------------------------------------")

# =====================================================
# 1️⃣ Configuration de base et Modèle (Passée à cl.on_chat_start)
# =====================================================

# Les fonctions utilitaires (save_to_long_term_memory) peuvent rester en dehors
def save_to_long_term_memory(thread_id: str, user_query: str, ai_response: str, vectordb_history_instance):
    """Enregistre la paire de messages dans la base Chroma pour la LTM."""
    content = f"Utilisateur ({thread_id}): {user_query}\nIA ({thread_id}): {ai_response}"
    vectordb_history_instance.add_texts([content], metadata={"thread_id": thread_id})


# =====================================================
# 2️⃣ Définition des Outils (Tool)
# =====================================================
# Les outils doivent être définis globalement ou transmis
load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("⚠️ GOOGLE_API_KEY manquante dans le fichier .env")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vectordb_rag_global = Chroma(embedding_function=embeddings, persist_directory="chroma_db")
vectordb_history_global = Chroma(embedding_function=embeddings, persist_directory="chroma_history")

@tool
def retrieve_context(query: str) -> str:
    """
    Récupère les documents les plus pertinents de la base de données 
    documentaire (RAG) pour répondre à une question factuelle.
    """
    results = vectordb_rag_global.similarity_search(query, k=3) 
    if not results:
        return "❌ Aucun document RAG pertinent trouvé."
    
    rag_context = ""
    source_documents = [] # Liste pour stocker les documents bruts pour l'affichage
    source_names_list = [] # Liste pour forcer le LLM à citer les sources

    for i, r in enumerate(results):
        source_path = r.metadata.get('source', f'Source RAG {i+1}')
        citation_name = f"source_{i+1}" # Nom Chainlit : source_1, source_2... (Nom simple pour la citation LLM)
        content = r.page_content
        
        # 🆕 ADAPTATION : Créer le nom d'affichage convivial
        try:
            # Tente de rendre le nom plus lisible (ex: enlever le chemin et reformater la page)
            base_name = os.path.basename(source_path)
            if ':page_' in base_name:
                # Ex: 'document.pdf:page_10' devient 'document.pdf - Page 10'
                display_name_friendly = base_name.replace(':page_', ' - Page ')
            else:
                # Si le format est juste le nom du fichier, utilise le nom et l'index du chunk
                display_name_friendly = f"{base_name} (Chunk {i+1})"
        except:
            display_name_friendly = f"Source {i+1} (Détails)"
        
        # 1. Stocke les documents bruts (pour l'affichage futur)
        source_documents.append({
            "content": content,
            "source": source_path,
            "name": citation_name, # Nom simple pour le lien cliquable
            "display_name": display_name_friendly # 👈 AJOUT DE LA CLÉ DISPLAY_NAME
        })
        
        # 2. Ajoute le nom de la source à la liste des citations
        source_names_list.append(citation_name)
        
        # 3. Construction du contexte RAG pour l'Agent
        rag_context += f"[DOCUMENT RAG {i+1} - CITATION: {citation_name}]: {content}\n---\n"
    
    # 4. ⚠️ STOCKAGE des DOCUMENTS bruts dans la session utilisateur
    cl.user_session.set("documents_to_display", source_documents) 
    
    # 5. Ajout d'une instruction forte pour forcer la citation des sources
    citation_instruction = f"\n\n**INSTRUCTION LLM:** Lorsque tu réponds à la question, utilise les sources ci-dessus et ajoute OBLIGATOIREMENT à la fin de ta réponse la liste des sources citées sous la forme : **Sources: {', '.join(source_names_list)}**."

    return rag_context + citation_instruction
@tool
def retrieve_history(query: str) -> str:
    """
    Récupère des fragments de conversations passées pertinentes de la mémoire 
    historique (chroma_history) pour le contexte à long terme.
    """
    results = vectordb_history_global.similarity_search(query, k=2) 
    
    if not results:
        return "❌ Aucun fragment de conversation passée pertinent trouvé."
    
    history_context = "\n\n".join(
        [f"💬 Fragment de conversation: {r.page_content[:200]}..." for r in results]
    )
    return history_context

tools = [retrieve_context, retrieve_history]


# =====================================================
# 3️⃣ Création de l'Agent (cl.on_chat_start)
# =====================================================

# Chainlit démarre l'agent ici et stocke les instances dans le "user session"
@cl.on_chat_start
async def start():
    
    # Modèle LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", 
        temperature=0.3,
    )

    
    # 🆕 Configuration du Checkpointer LangGraph avec chemin absolu
    # Notez le changement de syntaxe de la chaîne de connexion pour le chemin absolu
    checkpointer = InMemorySaver()

    system_prompt_text = (
        "Tu es un assistant RAG professionnel avec une mémoire à long terme. "
        "Tu as accès à deux outils : "
        "'retrieve_context' pour les faits documentaires généraux, "
        "'retrieve_history' pour te rappeler des discussions antérieures"
        " Utilise l'outil le plus approprié pour chaque requête. Réponds poliment."
    )
    
    # Création de l'Agent
    agent_instance = create_agent(
        llm, 
        tools, 
        system_prompt=system_prompt_text, 
        checkpointer=checkpointer, 
    )

    # Sauvegarde l'agent et la DB d'historique dans la session Chainlit
    cl.user_session.set("agent", agent_instance)
    cl.user_session.set("llm", llm) 
    cl.user_session.set("system_prompt_text", system_prompt_text) 
    cl.user_session.set("checkpointer", checkpointer) 
    
    cl.user_session.set("vectordb_history", vectordb_history_global)
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("documents_to_display", [])

    await cl.Message(content="Bonjour ! Je suis l'Agent RAG avec mémoire. Posez-moi votre première question.").send()


# =====================================================
# 4️⃣ Boucle de réponse (cl.on_message)
# =====================================================

@cl.on_message
async def main(message: cl.Message):
    # Récupération des objets depuis la session utilisateur
    vectordb_history_instance = cl.user_session.get("vectordb_history")
    thread_id_instance = cl.user_session.get("thread_id")
    agent_instance = cl.user_session.get("agent")
    query = message.content
    
    # -----------------------------------------------------------
    # ÉTAPE 1 : EXÉCUTION DE L'AGENT AVEC STREAMING
    # -----------------------------------------------------------
    
    checkpointer_config = {
        "configurable": {
            "thread_id": thread_id_instance 
        }
    }
    
    initial_messages = [HumanMessage(content=query)]
    
    # Nous envoyons un message temporaire pour signaler l'activité (optionnel)
    final_response_text = ""
    msg = cl.Message(content="🤖 Réflexion en cours...")
    await msg.send()

    try:
        # Lancement du stream de l'agent LangGraph
        async for token, metadata in agent_instance.astream( # 👈 Agent stocké
            {"messages": initial_messages}, 
            config=checkpointer_config,
            stream_mode="messages",
        ):
            message_token = token
            
            if isinstance(message_token, AIMessage) and message_token.content:
                await msg.stream_token(message_token.content)
                final_response_text += message_token.content
            
        # Finalisation de l'affichage Chainlit
        await msg.update() # 👈 Le message du LLM est maintenant complet

        # -----------------------------------------------------------
        # ÉTAPE 2 : PRÉPARATION ET ENVOI DU MESSAGE FINAL AVEC SOURCES
        # -----------------------------------------------------------

        # 1. Récupère les documents bruts stockés par l'outil
        source_documents = cl.user_session.get("documents_to_display")
        text_elements = []

        if source_documents:
            # Crée les cl.Text éléments (utilisant la nouvelle structure du doc)
            for doc in source_documents:

                # Récupère le nom convivial s'il existe, sinon utilise l'ancien format
                display_name_text = doc.get('display_name', f"{doc['name']} ({doc['source']})")

                text_elements.append(
                    cl.Text(
                        content=doc['content'], 
                        name=doc['name'], # 👈 Nom simple pour le lien (e.g., source_1)
                        display="side",
                        display_name=display_name_text # 👈 Nom du document/page
                    )
                )
            
            # 2. Attache les éléments au message final
            msg.elements = text_elements
            await msg.update() # 👈 Mise à jour finale pour afficher les sources

            # 3. Vider la liste après utilisation
            cl.user_session.set("documents_to_display", []) 

        # 4. Sauvegarde dans la LTM
        save_to_long_term_memory(thread_id_instance, query, final_response_text, vectordb_history_instance)
        print(f"💾 Conversation enregistrée dans 'chroma_history' (LTM) pour le thread {thread_id_instance}.")

    except Exception as e:
        error_message = (
            f"❌ Erreur critique LangGraph/Chainlit : {e}"
        )
        await cl.Message(content=error_message).send()
        print(error_message)