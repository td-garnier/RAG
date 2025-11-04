import subprocess

def ingest():
    print("🚀 Lancement de l’ingestion des documents...")
    # Appelle ton script RAG_ingest.py
    subprocess.run(["python", "RAG_ingest.py"], check=True)

def infer():
    print("💡 Lancement de l’inférence...")
    # Appelle ton script RAG_infer.py
    subprocess.run(["python", "RAG_infer.py"], check=True)

def main():
    print("Hello from envrag!")
    print("Choisis une action :")
    print("1️⃣ Ingestion")
    print("2️⃣ Inférence")

    choice = input("Entrez 1 ou 2 : ").strip()
    
    if choice == "1":
        ingest()
    elif choice == "2":
        infer()
    else:
        print("⚠️ Choix invalide. Fin du programme.")

if __name__ == "__main__":
    main()
