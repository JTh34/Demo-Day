#!/bin/bash

# Synchronization script: demoday_challenge -> puppycompanion-app
# Usage: ./sync_to_app.sh [--dry-run]

set -e

# Colors for messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Directories
DEV_DIR="./demoday_challenge"
APP_DIR="./puppycompanion-app"

# Display functions
log() {
    echo -e "${GREEN}[SYNC]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check that directories exist
if [ ! -d "$DEV_DIR" ]; then
    error "Development directory not found: $DEV_DIR"
    exit 1
fi

if [ ! -d "$APP_DIR" ]; then
    error "Application directory not found: $APP_DIR"
    exit 1
fi

# Dry-run mode
DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    warn "DRY-RUN mode enabled - no changes will be made"
fi

# Files to synchronize
FILES_TO_SYNC=(
    "agent_workflow.py"
    "rag_system.py"
    "embedding_models.py"
    "document_loader.py"
    "evaluation.py"
)

# Directories to synchronize
DIRS_TO_SYNC=(
    "data"
)

log "Starting synchronization..."

# Synchronize files
for file in "${FILES_TO_SYNC[@]}"; do
    if [ -f "$DEV_DIR/$file" ]; then
        if [ "$DRY_RUN" = true ]; then
            log "DRY-RUN: Would copy $file"
        else
            log "Synchronizing $file"
            cp "$DEV_DIR/$file" "$APP_DIR/"
        fi
    else
        warn "File not found: $DEV_DIR/$file"
    fi
done

# Synchronize directories (excluding Python 3.12 dev environment)
for dir in "${DIRS_TO_SYNC[@]}"; do
    if [ -d "$DEV_DIR/$dir" ]; then
        if [ "$DRY_RUN" = true ]; then
            log "DRY-RUN: Would synchronize directory $dir"
        else
            log "Synchronizing directory $dir"
            rsync -av --delete \
                --exclude='venv_dev_312/' \
                --exclude='__pycache__/' \
                --exclude='.ipynb_checkpoints/' \
                --exclude='*.pyc' \
                "$DEV_DIR/$dir/" "$APP_DIR/$dir/"
        fi
    else
        warn "Directory not found: $DEV_DIR/$dir"
    fi
done

# Check dependencies
if [ -f "$DEV_DIR/pyproject.toml" ] && [ -f "$APP_DIR/pyproject.toml" ]; then
    if ! diff -q "$DEV_DIR/pyproject.toml" "$APP_DIR/pyproject.toml" > /dev/null; then
        warn "pyproject.toml files differ. Please check dependencies."
        if [ "$DRY_RUN" = false ]; then
            log "Updating pyproject.toml"
            cp "$DEV_DIR/pyproject.toml" "$APP_DIR/"
        fi
    fi
fi

if [ "$DRY_RUN" = false ]; then
    log "Synchronization completed successfully!"
    log "Don't forget to test the application with: cd $APP_DIR && chainlit run app.py"
    warn "Note: Development environment (venv_dev_312) excluded from sync"
else
    log "DRY-RUN finished. Use './sync_to_app.sh' to perform the synchronization."
fi 