import os
import json
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# DeepEval imports
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from RAG_core import retrieve_context_core
from typing import List, Dict

# Custom models (votre code corrigé)
from generate_data import GeminiJudge
from RAG_agent2 import retrieve_context
from RAG_core import get_rag_documents
load_dotenv()

EVAL_MODEL_NAME = "gemini-2.5-flash-lite"  # Recommandé pour la précision du JSON
GEMINI_JUDGE = GeminiJudge(model_name=EVAL_MODEL_NAME)

# --- Fonction de Test de l'Application ---
# 🎯 Remplacez ceci par la fonction qui appelle votre agent RAG pour obtenir la réponse
def run_rag_agent(query: str, chat_history=None) -> Dict[str, any]:
    """
    Simule l'exécution de l'agent RAG (Récupération + Génération) en mode batch.
    Retourne la réponse finale et la liste des morceaux de contexte pour DeepEval.
    """
    
    # 1. Récupération des documents (utilise la logique de RAG_core.py)
    # ⚠️ Ceci renvoie une liste d'objets Document de LangChain
    raw_documents: List[Document] = get_rag_documents(query, k=4)
    
    # 2. Préparation du contexte pour le LLM et pour DeepEval
    retrieved_context_list = [doc.page_content for doc in raw_documents] # Liste des morceaux bruts pour DeepEval
    context_for_llm = "\n---\n".join(retrieved_context_list) # Chaîne unique pour le Prompt LLM
    
    # 3. Génération (Simule l'étape finale de l'Agent)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # Prompt System/Template (simplifié par rapport à LangGraph mais fonctionnel)
    system_prompt = (
        "Tu es un assistant RAG professionnel. Utilise UNIQUEMENT le contexte ci-dessous pour répondre à la question. "
        "Si tu ne trouves pas la réponse, réponds poliment que l'information n'est pas disponible dans tes documents."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nCONTEXTE: {context}"),
        ("human", "{query}")
    ])
    
    chain = prompt | llm
    
    # Exécution de la chaîne
    final_response = chain.invoke({
        "context": context_for_llm,
        "query": query
    }).content
    
    return {
        "actual_output": final_response,
        # 🎯 CRUCIAL : DeepEval a besoin de la liste des chunks
        "retrieved_context": retrieved_context_list 
    }


# --- SCRIPT PRINCIPAL D'ÉVALUATION ---
def main_evaluation():
    DATA_FILE = "synthetic_rag_test_data.json"
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ Fichier de données non trouvé : {DATA_FILE}. Veuillez lancer 'uv run generate_data.py' d'abord.")
        return

    print("⏳ Chargement des cas de test...")
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        synthetic_data = json.load(f)

    test_cases = []
    
    # 1. Préparer les cas de test
    for item in synthetic_data:
        # 2. Lancer l'agent pour obtenir la réponse réelle et le contexte
        agent_result = run_rag_agent(item['query'])
        
        # 3. Créer le LLMTestCase requis par DeepEval
        test_case = LLMTestCase(
            input=item['query'],
            actual_output=agent_result['actual_output'],
            expected_output=item['expected_output'],
            # 🎯 CORRECTION : Utiliser 'retrieval_context' au lieu de 'context'
            retrieval_context=agent_result['retrieved_context'] 
        )
        test_cases.append(test_case)

    # 4. Définir les métriques RAG
    print(f"🚀 Démarrage de l'évaluation sur {len(test_cases)} cas de test...")
    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=GEMINI_JUDGE,async_mode=False), #Evaluates if the generated answer is relevant to the user query
        FaithfulnessMetric(threshold=0.8, model=GEMINI_JUDGE,async_mode=False) # Measures if the generated answer is factually consistent with the provided context
    ]

    # 5. Lancer l'évaluation DeepEval
    evaluation_result = evaluate(test_cases=test_cases, metrics=metrics)

    # 6. Sauvegarder les résultats (méthode manuelle robuste)
    OUTPUT_DIR = "evaluation_reports"
    REPORT_FILE = os.path.join(OUTPUT_DIR, "evaluation_report.json")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Extraire les résultats au format Pydantic/JSON
    # La méthode .to_dict() ou .model_dump() fonctionne sur les objets Pydantic (ce qu'est EvaluationResult)
    try:
        results_data = evaluation_result.model_dump() # Pour les versions DeepEval utilisant Pydantic v2
    except AttributeError:
        results_data = evaluation_result.to_dict() # Pour les versions DeepEval utilisant Pydantic v1
        
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Résultats de l'évaluation sauvegardés dans le dossier : {OUTPUT_DIR}/")


if __name__ == "__main__":
    main_evaluation()