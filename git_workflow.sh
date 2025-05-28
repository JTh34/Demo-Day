#!/bin/bash

# Script de gestion Git pour PuppyCompanion
# Usage: ./git_workflow.sh [option]

set -e

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction d'affichage coloré
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Fonction pour afficher l'aide
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  status      Afficher le statut Git"
    echo "  add         Ajouter tous les fichiers modifiés"
    echo "  commit      Commit interactif avec message personnalisé"
    echo "  quick       Commit rapide avec message automatique"
    echo "  push        Pousser vers le dépôt distant"
    echo "  pull        Récupérer les changements du dépôt distant"
    echo "  log         Afficher l'historique des commits"
    echo "  branch      Gérer les branches"
    echo "  sync        Synchroniser avec le dépôt distant"
    echo "  backup      Créer un commit de sauvegarde"
    echo "  help        Afficher cette aide"
    echo ""
    echo "Exemples:"
    echo "  $0 status"
    echo "  $0 commit"
    echo "  $0 quick"
    echo "  $0 backup"
}

# Fonction pour vérifier si on est dans un dépôt Git
check_git_repo() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Pas dans un dépôt Git!"
        exit 1
    fi
}

# Fonction pour afficher le statut
git_status() {
    print_status "Statut du dépôt Git:"
    git status --short
    echo ""
    git log --oneline -5
}

# Fonction pour ajouter les fichiers
git_add() {
    print_status "Ajout des fichiers modifiés..."
    git add .
    print_success "Fichiers ajoutés avec succès"
    git status --short
}

# Fonction pour commit interactif
git_commit() {
    # Vérifier s'il y a des changements à commiter
    if git diff --cached --quiet; then
        print_warning "Aucun changement à commiter. Utilisez 'add' d'abord."
        return 1
    fi
    
    echo "Fichiers à commiter:"
    git diff --cached --name-only
    echo ""
    
    read -p "Message de commit: " commit_message
    
    if [ -z "$commit_message" ]; then
        print_error "Message de commit requis!"
        return 1
    fi
    
    git commit -m "$commit_message"
    print_success "Commit créé avec succès"
}

# Fonction pour commit rapide
git_quick_commit() {
    # Ajouter tous les fichiers
    git add .
    
    # Vérifier s'il y a des changements
    if git diff --cached --quiet; then
        print_warning "Aucun changement à commiter."
        return 1
    fi
    
    # Générer un message automatique basé sur les fichiers modifiés
    modified_files=$(git diff --cached --name-only | head -5)
    timestamp=$(date "+%Y-%m-%d %H:%M")
    
    commit_message="Auto-commit: Updates at $timestamp"
    
    git commit -m "$commit_message"
    print_success "Commit rapide créé: $commit_message"
}

# Fonction pour créer un commit de sauvegarde
git_backup() {
    git add .
    
    if git diff --cached --quiet; then
        print_warning "Aucun changement à sauvegarder."
        return 1
    fi
    
    timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
    backup_message="Backup: $timestamp"
    
    git commit -m "$backup_message"
    print_success "Sauvegarde créée: $backup_message"
}

# Fonction pour gérer les branches
git_branch() {
    echo "Branches locales:"
    git branch
    echo ""
    echo "Branches distantes:"
    git branch -r
    echo ""
    read -p "Créer une nouvelle branche? (y/N): " create_branch
    
    if [[ $create_branch =~ ^[Yy]$ ]]; then
        read -p "Nom de la nouvelle branche: " branch_name
        if [ ! -z "$branch_name" ]; then
            git checkout -b "$branch_name"
            print_success "Branche '$branch_name' créée et activée"
        fi
    fi
}

# Fonction pour synchroniser avec le dépôt distant
git_sync() {
    print_status "Synchronisation avec le dépôt distant..."
    
    # Vérifier s'il y a un dépôt distant configuré
    if ! git remote | grep -q origin; then
        print_warning "Aucun dépôt distant configuré."
        read -p "Ajouter un dépôt distant? (y/N): " add_remote
        
        if [[ $add_remote =~ ^[Yy]$ ]]; then
            read -p "URL du dépôt distant: " remote_url
            if [ ! -z "$remote_url" ]; then
                git remote add origin "$remote_url"
                print_success "Dépôt distant ajouté"
            fi
        fi
        return
    fi
    
    git fetch origin
    git pull origin main
    print_success "Synchronisation terminée"
}

# Fonction principale
main() {
    check_git_repo
    
    case "${1:-help}" in
        "status")
            git_status
            ;;
        "add")
            git_add
            ;;
        "commit")
            git_commit
            ;;
        "quick")
            git_quick_commit
            ;;
        "push")
            git push origin main
            print_success "Changements poussés vers le dépôt distant"
            ;;
        "pull")
            git pull origin main
            print_success "Changements récupérés du dépôt distant"
            ;;
        "log")
            git log --oneline --graph -10
            ;;
        "branch")
            git_branch
            ;;
        "sync")
            git_sync
            ;;
        "backup")
            git_backup
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Exécuter la fonction principale avec tous les arguments
main "$@" 