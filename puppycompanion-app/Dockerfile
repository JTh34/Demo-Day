FROM python:3.10-slim

WORKDIR /app

# Copier d'abord les fichiers de dépendances
COPY pyproject.toml README.md ./

# Installer les dépendances avec les versions actualisées
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir chainlit==1.0.101 python-dotenv

# Vérifier que langchain-qdrant est correctement installé
RUN python -c "from langchain_qdrant import Qdrant; print('Qdrant correctement importé')"

# Créer les répertoires nécessaires avec les bonnes permissions
RUN mkdir -p /app/.files && \
    chmod -R 777 /app/.files

# Créer le répertoire pour le stockage Qdrant
RUN mkdir -p /tmp/qdrant_storage && \
    chmod -R 777 /tmp/qdrant_storage

# Copier tous les autres fichiers 
COPY . .

# Supprimer le répertoire .chainlit s'il existe pour permettre à Chainlit de le régénérer
RUN rm -rf /app/.chainlit

# S'assurer que tous les fichiers et répertoires ont les bonnes permissions
RUN chmod -R 755 /app
RUN mkdir -p /app/.chainlit && chmod -R 777 /app/.chainlit

# Exposer le port
EXPOSE 7860

# Configurer des variables d'environnement pour Chainlit
ENV CHAINLIT_ROOT_PATH=""
ENV CHAINLIT_SERVER_URL="/"
ENV PORT=7860
ENV CHAINLIT_MAX_SIZE_MB=100
# Spécifier explicitement le répertoire des fichiers
ENV CHAINLIT_FILES_DIRECTORY="/app/.files"

# Lancer l'application
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860", "--headless"]