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
from RAG_core import retrieve_context_core, vectordb_rag_global, vectordb_history_global

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
# 1️⃣ Configuration de base et Modèle
# =====================================================

# fonctions utilitaires pour sauvegarder dans la LTM
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
    # 🎯 1. Appel de la logique RAG CORE pour obtenir le contexte formaté
    context_for_llm = retrieve_context_core(query) 

    # 2. Logiciel d'affichage Chainlit : Récupérer les documents bruts pour cl.Pdf
    
    # Utilisez la DB globale importée de RAG_core
    retriever_rag = vectordb_rag_global.as_retriever(search_kwargs={"k": 4}) 
    source_documents = retriever_rag.invoke(query) 

    documents_for_display = []
    PDF_FOLDER = "./documents" 

    for doc in source_documents:
        # Créer les objets nécessaires pour cl.user_session.set()
        source_name_only = doc.metadata.get('source', 'Inconnu')
        pdf_path = os.path.join(PDF_FOLDER, source_name_only)
        page_number_for_display = doc.metadata.get('page_label', 1)
        citation_name = f"source_{len(documents_for_display) + 1}"
        display_name_friendly = f"{source_name_only} (Page {page_number_for_display})"
        
        documents_for_display.append({
            "content": doc.page_content,
            "source": source_name_only, 
            "name": citation_name,
            "display_name": display_name_friendly,
            "path": pdf_path,
            "page": page_number_for_display
        })

    # 3. Mise à jour de la session Chainlit
    cl.user_session.set("documents_to_display", documents_for_display)

    return context_for_llm # Renvoyer le contexte formaté par la fonction CORE

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
        temperature=0,
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
        msg.content = ""
        await msg.update()

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
        elements_to_attach = []

        if source_documents:
            for doc in source_documents:
                # Vérifie si le chemin d'accès au PDF est valide et existe
                if os.path.exists(doc.get('path')):
                    
                    elements_to_attach.append(
                        # 🆕 Utilisation de cl.Pdf
                        cl.Pdf(
                            name=doc['name'], 
                            display="side", # Affichage dans le panneau latéral
                            path=doc['path'], # Chemin local du fichier PDF
                            page=doc.get('page', 1), # Page à laquelle ouvrir le PDF (par défaut 1)
                            display_name=doc.get('display_name', doc['name']) # Nom convivial
                        )
                    )
                else:
                    # Si le PDF n'est pas trouvé (erreur dans le chemin), 
                    # on affiche au moins le chunk textuel comme secours
                    elements_to_attach.append(
                        cl.Text(
                            content=doc['content'], 
                            name=doc['name'],
                            display="side",
                            display_name=doc.get('display_name', doc['name'])
                        )
                    )
            
            # 2. Attache les éléments (maintenant des cl.Pdf) au message final
            msg.elements = elements_to_attach
            await msg.update()

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