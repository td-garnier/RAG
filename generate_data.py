import os
import json
import time # Pour la méthode synchrone time.sleep()

# 🆕 Imports corrigés
from deepeval.synthesizer import Synthesizer 
from deepeval.models import DeepEvalBaseLLM
from deepeval.dataset import EvaluationDataset
from google import genai
from google.genai.errors import APIError
from dotenv import load_dotenv
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.models import DeepEvalBaseEmbeddingModel
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# =========================================================================
# --- CLASSE WRAPPER POUR EMBEDDING HUGGING FACE (EXPOSÉE POUR L'IMPORT) ---
# =========================================================================
class HFEmbedder(DeepEvalBaseEmbeddingModel):
    # ... (Le corps de la classe reste inchangé) ...
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)
    
    def load_model(self):
        pass

    def get_model_name(self):
        return self.model_name
    
    def embed_text(self, text: str):
        return self.embedder.embed_query(text)

    def embed_texts(self, texts: list[str]):
        return self.embedder.embed_documents(texts) 

    async def a_embed_text(self, text: str):
        return self.embed_text(text)

    async def a_embed_texts(self, texts: list[str]):
        return self.embed_texts(texts)


# ===============================================================
# --- Classe Custom LLM pour DeepEval (GeminiJudge) (EXPOSÉE) ---
# ===============================================================
class GeminiJudge(DeepEvalBaseLLM):
    REQUEST_DELAY = 10
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        self._model_name = model_name
        try:
            load_dotenv()
            if not os.environ.get("GOOGLE_API_KEY"):
                 raise ValueError("GOOGLE_API_KEY non définie.")
            self.client = genai.Client()
        except Exception as e:
            print(f"❌ Erreur d'initialisation GeminiJudge: {e}")
            raise

    def load_model(self):
        pass

    def get_model_name(self):
        return self._model_name

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config={"temperature": 0.0}
            )
            time.sleep(self.REQUEST_DELAY)
            return response.text
        except APIError as e:
            print(f"Erreur d'API Gemini lors de la génération: {e}")
            return "Erreur lors de la génération."

# =========================================================================
# --- LOGIQUE D'EXÉCUTION DE LA GÉNÉRATION (Protégée) ---
# =========================================================================

# --- Configuration commune ---
# Ces variables doivent être définies AVANT main_generation
gemini_judge = GeminiJudge(model_name="gemini-2.5-flash-lite")
DOCUMENT_PATHS = ["./documents/SV25-FR.pdf", "./documents/airfryer.pdf"] 
hf_embedder = HFEmbedder()
NUM_TESTS = 5 
OUTPUT_FILE = "synthetic_rag_test_data.json"


def main_generation():
    """Exécute la génération de cas de test synthétiques avec DeepEval Synthesizer."""

    print("⏳ Démarrage de la génération de cas de test synthétiques avec Gemini/Synthesizer...")

    try:
        # 1. Configurer le Context Construction
        context_config = ContextConstructionConfig(
            critic_model=gemini_judge, 
            embedder=hf_embedder,       
            max_contexts_per_document=2 
        )

        # 2. Utilisation du Synthesizer
        synthesizer = Synthesizer(
            model=gemini_judge 
        )
        
        # 3. Appel de la méthode avec la nouvelle configuration
        goldens = synthesizer.generate_goldens_from_docs(
            document_paths=DOCUMENT_PATHS,
            max_goldens_per_context=2,
            context_construction_config=context_config 
        )

        # Sauvegarde des données dans un fichier JSON réutilisable
        test_data_list = []
        for golden in goldens:
            test_data_list.append({
                "query": golden.input,
                "expected_output": golden.expected_output,
                "context": golden.context
            })
            
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(test_data_list, f, indent=4, ensure_ascii=False)

        print(f"\n✅ Génération terminée. {len(goldens)} cas de test (Goldens) enregistrés dans {OUTPUT_FILE}")

    except Exception as e:
        print(f"\n❌ Erreur lors de la génération des cas de test : {e}")

# --- PROTECTION CONTRE L'EXÉCUTION LORS DE L'IMPORTATION ---
if __name__ == "__main__":
    main_generation()