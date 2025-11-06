import subprocess
import os
import sys

# La cible est RAG_agent2.py, là où se trouve l'application Chainlit
RAG_AGENT_FILE = "RAG_agent2.py"

def ingest():
    print("🚀 Lancement de l’ingestion des documents...")
    # Appelle ton script RAG_ingest.py
    # NOTE: Assurez-vous que RAG_ingest.py ne lance PAS de serveur ou n'utilise PAS Chainlit
    subprocess.run(["python", "RAG_ingest.py"], check=True)

def infer():
    print("💡 Lancement de l’inférence Chainlit...")
    
    # IMPORTANT : Utilise `uv run chainlit run` pour lancer RAG_agent2.py
    # La librairie Chainlit lance le serveur web qui exécute les fonctions décorées.
    try:
        # Nous utilisons 'uv run' pour exécuter la commande Chainlit dans l'environnement virtuel
        # Le -w est pour le mode watch (rechargement automatique)
        # La commande complète devient: uv run chainlit run RAG_agent2.py -w
        subprocess.run(["chainlit", "run", RAG_AGENT_FILE, "-w"], check=True)
        
    except FileNotFoundError:
        print("❌ Erreur: Chainlit n'est pas trouvé. Assurez-vous que Chainlit est installé et que 'uv run' fonctionne.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du lancement de Chainlit: {e}")

def main():
    print("Hello from envrag!")
    print("Choisis une action :")
    print("1️⃣ Ingestion")
    print("2️⃣ Inférence (Lancement Chainlit)")

    choice = input("Entrez 1 ou 2 : ").strip()
    
    if choice == "1":
        ingest()
    elif choice == "2":
        # Nous utilisons maintenant `infer` pour lancer Chainlit
        infer()
    else:
        print("⚠️ Choix invalide. Fin du programme.")

if __name__ == "__main__":
    main()