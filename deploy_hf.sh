#!/bin/bash

# HuggingFace deployment script for PuppyCompanion
# Usage: ./deploy_hf.sh [--space-name SPACE_NAME]

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
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

# Default configuration
SPACE_NAME=""
APP_DIR="./puppycompanion-app"
HF_USERNAME="JTh34"  # Replace with your HF username

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --space-name)
            SPACE_NAME="$2"
            shift 2
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ask for space name if not provided
if [ -z "$SPACE_NAME" ]; then
    read -p "HuggingFace space name (e.g. puppycompanion-v2): " SPACE_NAME
fi

if [ -z "$SPACE_NAME" ]; then
    error "Space name required"
    exit 1
fi

# Check that the app directory exists
if [ ! -d "$APP_DIR" ]; then
    error "Application directory not found: $APP_DIR"
    exit 1
fi

cd "$APP_DIR"

log "Preparing deployment for space: $HF_USERNAME/$SPACE_NAME"

# Pre-deployment checks
log "Pre-deployment checks..."

# Check required files
REQUIRED_FILES=(
    "app.py"
    "Dockerfile"
    "README.md"
    "data/preprocessed_chunks.json"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        error "Missing required file: $file"
        exit 1
    fi
done

# Check that API keys are not in files
if [ -f ".env" ]; then
    if grep -q "sk-" .env 2>/dev/null; then
        error "API keys detected in .env - do not include them in deployment!"
        exit 1
    fi
fi

# Create requirements.txt from pyproject.toml if needed
if [ ! -f "requirements.txt" ] && [ -f "pyproject.toml" ]; then
    log "Generating requirements.txt..."
    if command -v pip-compile &> /dev/null; then
        pip-compile pyproject.toml
    else
        warn "pip-compile not found. Basic requirements.txt generation..."
        # Basic extraction of dependencies from pyproject.toml
        if command -v python &> /dev/null; then
            python -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
deps = data.get('project', {}).get('dependencies', [])
with open('requirements.txt', 'w') as f:
    for dep in deps:
        f.write(dep + '\n')
" 2>/dev/null || {
                warn "Could not generate requirements.txt automatically"
                warn "Please create the requirements.txt file manually"
                exit 1
            }
        fi
    fi
fi

# Check SSH connection to HuggingFace
log "Checking HuggingFace SSH connection..."
if ! ssh -T git@hf.co -o ConnectTimeout=10 -o BatchMode=yes 2>/dev/null; then
    warn "SSH connection to HuggingFace failed"
    info "Check your SSH configuration or use: ssh-keygen -t ed25519 -C 'your_email@example.com'"
    info "Then add the public key to your HuggingFace profile"
    
    # Fallback to HTTPS if available
    if command -v huggingface-cli &> /dev/null && huggingface-cli whoami &> /dev/null; then
        warn "Using HTTPS as fallback..."
        USE_SSH=false
    else
        error "No authentication method available"
        exit 1
    fi
else
    log "HuggingFace SSH connection OK"
    USE_SSH=true
fi

# Create the space if it does not exist (requires huggingface-cli)
if command -v huggingface-cli &> /dev/null; then
    log "Creating/updating HuggingFace space..."
    huggingface-cli repo create "$SPACE_NAME" --type space --space_sdk docker 2>/dev/null || true
else
    warn "huggingface-cli not available, make sure the space exists"
fi

# Clone or update the repo
TEMP_DIR="/tmp/hf_deploy_$$"
log "Cloning repository..."

if [ "$USE_SSH" = true ]; then
    # Use SSH
    git clone "git@hf.co:spaces/$HF_USERNAME/$SPACE_NAME" "$TEMP_DIR" || {
        error "SSH clone failed. Check username and space name."
        exit 1
    }
else
    # Use HTTPS
    git clone "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME" "$TEMP_DIR" || {
        error "HTTPS clone failed. Check username and space name."
        exit 1
    }
fi

# Copy files
log "Copying files..."
rsync -av --exclude='.git' --exclude='venv_*' --exclude='.env' --exclude='__pycache__' --exclude='.chainlit' ./ "$TEMP_DIR/"

# Go to temp directory
cd "$TEMP_DIR"

# Configure Git if needed
git config user.email "action@github.com" || true
git config user.name "Deploy Script" || true

# Add and commit
log "Committing changes..."
git add .
git commit -m "Deploy PuppyCompanion $(date '+%Y-%m-%d %H:%M:%S')" || {
    warn "No changes detected"
}

# Push to HuggingFace
log "Pushing to HuggingFace..."
git push

# Clean up
cd - > /dev/null
rm -rf "$TEMP_DIR"

log "Deployment completed successfully!"
info "Your app will be available at: https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
info "Deployment may take a few minutes..." 