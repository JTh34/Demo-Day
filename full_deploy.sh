#!/bin/bash

# Script complet de déploiement PuppyCompanion
# Usage: ./full_deploy.sh [--space-name SPACE_NAME] [--skip-sync] [--skip-test]

set -e

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[DEPLOY]${NC} $1"
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

step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# Configuration par défaut
SPACE_NAME=""
SKIP_SYNC=false
SKIP_TEST=false

# Parser les arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --space-name)
            SPACE_NAME="$2"
            shift 2
            ;;
        --skip-sync)
            SKIP_SYNC=true
            shift
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --space-name NAME    Nom de l'espace HuggingFace"
            echo "  --skip-sync         Ignorer la synchronisation"
            echo "  --skip-test         Ignorer les tests locaux"
            echo "  --help              Afficher cette aide"
            exit 0
            ;;
        *)
            error "Option inconnue: $1"
            echo "Utilisez --help pour voir les options disponibles"
            exit 1
            ;;
    esac
done

echo "🚀 Déploiement Complet PuppyCompanion"
echo "======================================"

# Étape 1: Synchronisation
if [ "$SKIP_SYNC" = false ]; then
    step "1/4 - Synchronisation des modifications"
    
    # Vérifier d'abord ce qui sera synchronisé
    log "Vérification des modifications à synchroniser..."
    ./sync_to_app.sh --dry-run
    
    echo
    read -p "Continuer avec la synchronisation ? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./sync_to_app.sh
    else
        warn "Synchronisation annulée par l'utilisateur"
        exit 1
    fi
else
    step "1/4 - Synchronisation ignorée (--skip-sync)"
fi

echo

# Étape 2: Tests locaux
if [ "$SKIP_TEST" = false ]; then
    step "2/4 - Tests locaux"
    
    log "Lancement des tests locaux..."
    echo "Les tests vont démarrer l'application. Fermez-la avec Ctrl+C quand vous aurez terminé vos tests."
    echo
    read -p "Continuer avec les tests ? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./test_app.sh || {
            error "Tests locaux échoués"
            exit 1
        }
    else
        warn "Tests locaux ignorés par l'utilisateur"
    fi
else
    step "2/4 - Tests locaux ignorés (--skip-test)"
fi

echo

# Étape 3: Vérification SSH
step "3/4 - Vérification de la configuration HuggingFace"

log "Test de la configuration SSH..."
if ./test_hf_ssh.sh; then
    log "Configuration SSH validée ✅"
else
    error "Configuration SSH échouée"
    exit 1
fi

echo

# Étape 4: Déploiement
step "4/4 - Déploiement sur HuggingFace"

if [ -n "$SPACE_NAME" ]; then
    log "Déploiement vers l'espace: $SPACE_NAME"
    ./deploy_hf.sh --space-name "$SPACE_NAME"
else
    log "Déploiement interactif..."
    ./deploy_hf.sh
fi

echo
log "🎉 Déploiement complet terminé avec succès!"
info "Votre application PuppyCompanion est maintenant déployée sur HuggingFace Spaces"

# Résumé des étapes
echo
echo "📋 Résumé des étapes effectuées:"
if [ "$SKIP_SYNC" = false ]; then
    echo "  ✅ Synchronisation des modifications"
else
    echo "  ⏭️  Synchronisation ignorée"
fi

if [ "$SKIP_TEST" = false ]; then
    echo "  ✅ Tests locaux"
else
    echo "  ⏭️  Tests locaux ignorés"
fi

echo "  ✅ Vérification SSH HuggingFace"
echo "  ✅ Déploiement sur HuggingFace Spaces"

echo
info "Prochaines étapes recommandées:"
info "1. Vérifiez que l'application fonctionne sur HuggingFace"
info "2. Testez les fonctionnalités principales"
info "3. Surveillez les logs pour détecter d'éventuels problèmes" 