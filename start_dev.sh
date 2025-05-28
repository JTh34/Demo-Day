#!/bin/bash

# Script de lancement rapide pour l'environnement de développement
# Usage: ./start_dev.sh

set -e

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[DEV]${NC} $1"
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

# Répertoires
DEV_DIR="./demoday_challenge"
VENV_DEV="venv_dev_312"

log "Lancement de l'environnement de développement Python 3.12..."

# Vérifier que le répertoire existe
if [ ! -d "$DEV_DIR" ]; then
    error "Répertoire de développement non trouvé: $DEV_DIR"
    exit 1
fi

# Vérifier que l'environnement virtuel existe
if [ ! -d "$DEV_DIR/$VENV_DEV" ]; then
    error "Environnement virtuel Python 3.12 non trouvé: $DEV_DIR/$VENV_DEV"
    echo
    info "Configurez d'abord l'environnement avec:"
    echo "  ./setup_dev_env.sh"
    exit 1
fi

# Aller dans le répertoire de développement
cd "$DEV_DIR"

# Activer l'environnement virtuel
log "Activation de l'environnement Python 3.12..."
source "$VENV_DEV/bin/activate"

# Vérifier la version Python
PYTHON_VERSION=$(python --version)
log "Version Python active: $PYTHON_VERSION"

# Vérifier que Jupyter est installé
if ! command -v jupyter &> /dev/null; then
    error "Jupyter non trouvé dans l'environnement"
    echo
    info "Réinstallez l'environnement avec:"
    echo "  ./setup_dev_env.sh"
    exit 1
fi

# Lancer Jupyter Lab
log "Lancement de Jupyter Lab..."
info "Sélectionnez le kernel: 'PuppyCompanion Dev (Python 3.12)'"
echo
warn "Pour arrêter Jupyter: Ctrl+C dans ce terminal"
echo

# Lancer Jupyter Lab en arrière-plan et ouvrir le navigateur
jupyter lab --no-browser --port=8888 &
JUPYTER_PID=$!

# Attendre un peu que Jupyter démarre
sleep 3

# Ouvrir le navigateur (macOS)
if command -v open &> /dev/null; then
    open "http://localhost:8888"
fi

log "Jupyter Lab démarré sur http://localhost:8888"
log "PID du processus Jupyter: $JUPYTER_PID"

# Attendre que l'utilisateur arrête Jupyter
wait $JUPYTER_PID 