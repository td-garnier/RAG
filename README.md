# RAG Project

Ce projet permet de **ingérer des documents PDF** et de **faire de l'inférence** via un système RAG (Retrieval-Augmented Generation) utilisant Chroma et Gemini.

---

## Prérequis

- Python 3.13
- [Google API Key](https://developers.google.com/identity) pour accéder à Gemini et créer un fichier .env avec la clé "GOOGLE_API_KEY="
- Les fichiers PDF à analyser dans le dossier `./documents`

```bash

```

## Structure du projet

RAG_proj/
├─ documents/        # PDF à ingérer

├─ .env/        # variable d'environnement

├─ chroma_db/        # Base vectorielle persistée

├─ RAG_ingest.py     # Script d'ingestion

├─ RAG_infer.py      # Script d'inférence

├─ main.py           # Interface principale pour choisir ingestion ou inférence

├─ requirements.txt  # Dépendances Python

└─ README.md

## Lancer le projet

Depuis le dossier RAG_proj :

```bash
uv run main.py
```

### Ingestion de documents

* Parcourt le dossier `documents/` pour trouver les PDFs non indexés.
* Extrait le texte avec OCR si nécessaire.
* Crée ou met à jour la base vectorielle Chroma dans `chroma_db/`.
* Génère des tags via Google Gemini pour chaque document.

### Inférence / Questions

* Permet de poser des questions sur les documents indexés.
* Le système retourne des réponses contextuelles basées sur la base vectorielle et Gemini.

---

## Exemple d'utilisation en inference

```bash
uv run main.py
infer choisi
> Quelle est la puissance de l'airfryer ?
Réponse : 1500W
```

---

## Notes

* Les fichiers PDF doivent être placés dans `documents/`.
* La base Chroma est persistée dans `chroma_db/`.
* Pour réinitialiser l’environnement, vous pouvez recréer le venv `uv` et réinstaller les dépendances.
