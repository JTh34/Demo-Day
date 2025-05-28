#!/bin/bash

# Script de configuration de l'environnement de développement
# Spécial Apple Silicon M4 + Python 3.12 pour unstructured.io
# Usage: ./setup_dev_env.sh

set -e

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[SETUP]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Vérifier si on est sur Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    warn "Ce script est optimisé pour Apple Silicon (M1/M2/M3/M4)"
fi

# Répertoires
DEV_DIR="./demoday_challenge"
VENV_DEV="venv_dev_312"

log "Configuration de l'environnement de développement Python 3.12..."

# Vérifier Python 3.12
if ! command -v python3.12 &> /dev/null; then
    error "Python 3.12 non trouvé. Installez-le avec:"
    echo "  brew install python@3.12"
    echo "  ou pyenv install 3.12.0"
    exit 1
fi

# Créer l'environnement virtuel Python 3.12
if [ ! -d "$DEV_DIR/$VENV_DEV" ]; then
    log "Création de l'environnement virtuel Python 3.12..."
    cd "$DEV_DIR"
    python3.12 -m venv "$VENV_DEV"
    cd ..
else
    log "Environnement virtuel Python 3.12 existant trouvé"
fi

# Activer et installer les dépendances
log "Installation des dépendances pour le développement..."
cd "$DEV_DIR"
source "$VENV_DEV/bin/activate"

# Mise à jour pip
pip install --upgrade pip

# Installation des dépendances principales directement
log "Installation des dépendances principales..."
pip install \
    "langchain>=0.0.300" \
    "langchain-community>=0.0.16" \
    "langchain-core>=0.1.0" \
    "langchain-openai>=0.0.5" \
    "langchain-qdrant>=0.0.1" \
    "qdrant-client>=1.6.0" \
    "chainlit>=1.0.0" \
    "openai>=1.6.0" \
    "python-dotenv>=1.0.0" \
    "tavily-python>=0.2.4" \
    "pandas>=2.0.0" \
    "numpy>=1.24.0" \
    "tqdm>=4.66.0"

# Installation spécifique pour Apple Silicon M4
log "Installation d'unstructured.io pour Apple Silicon M4..."
pip install "unstructured[local-inference]>=0.11.0"

# Installation des outils de développement
log "Installation des outils Jupyter..."
pip install jupyter jupyterlab ipykernel

# Installation d'outils supplémentaires pour le développement
log "Installation d'outils de développement supplémentaires..."
pip install \
    "matplotlib>=3.7.0" \
    "seaborn>=0.12.0" \
    "pymupdf>=1.22.0" \
    "pypdf>=3.15.1" \
    "pillow>=10.0.0" \
    "nest-asyncio>=1.5.6"

# Enregistrer le kernel Jupyter
log "Enregistrement du kernel Jupyter Python 3.12..."
python -m ipykernel install --user --name="puppycompanion-dev-312" --display-name="PuppyCompanion Dev (Python 3.12)"

cd ..

log "Configuration terminée avec succès!"
echo
info "Pour utiliser l'environnement de développement:"
echo "  cd demoday_challenge/"
echo "  source venv_dev_312/bin/activate"
echo "  jupyter lab"
echo
info "Dans Jupyter, sélectionnez le kernel: 'PuppyCompanion Dev (Python 3.12)'"
echo
warn "IMPORTANT: Utilisez cet environnement UNIQUEMENT pour le développement/notebooks"
warn "L'environnement de production reste en Python 3.11 dans puppycompanion-app/" 