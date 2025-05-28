#!/bin/bash

# HuggingFace SSH configuration test script
# Usage: ./test_hf_ssh.sh

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[SSH-TEST]${NC} $1"
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

log "Testing HuggingFace SSH configuration..."

# Check if SSH is installed
if ! command -v ssh &> /dev/null; then
    error "SSH is not installed"
    exit 1
fi

# Check for SSH keys
log "Checking for SSH keys..."
if [ ! -d "$HOME/.ssh" ]; then
    warn "Directory ~/.ssh not found"
    info "Create an SSH key with: ssh-keygen -t ed25519 -C 'your_email@example.com'"
    exit 1
fi

# List available keys
SSH_KEYS=$(find "$HOME/.ssh" -name "*.pub" 2>/dev/null || true)
if [ -z "$SSH_KEYS" ]; then
    warn "No public SSH key found"
    info "Create an SSH key with: ssh-keygen -t ed25519 -C 'your_email@example.com'"
    exit 1
else
    log "SSH keys found:"
    echo "$SSH_KEYS" | while read -r key; do
        echo "  - $key"
    done
fi

# Test connection to HuggingFace
log "Testing connection to HuggingFace..."
if ssh -T git@hf.co -o ConnectTimeout=10 -o BatchMode=yes 2>/dev/null; then
    log "✅ SSH connection to HuggingFace successful!"
    
    # Test a Git operation
    log "Testing a Git operation..."
    TEMP_DIR="/tmp/hf_ssh_test_$$"
    
    # Try to clone a public repo for testing
    if git clone git@hf.co:spaces/huggingface/README "$TEMP_DIR" 2>/dev/null; then
        log "✅ SSH Git clone successful!"
        rm -rf "$TEMP_DIR"
    else
        warn "⚠️  SSH connection OK but Git clone failed"
        info "This may be normal if you do not have access to this repo"
    fi
    
else
    error "❌ SSH connection to HuggingFace failed"
    echo
    info "Possible solutions:"
    info "1. Check that your SSH key is added to your HuggingFace profile:"
    info "   https://huggingface.co/settings/keys"
    echo
    info "2. If you do not have an SSH key, create one:"
    info "   ssh-keygen -t ed25519 -C 'your_email@example.com'"
    echo
    info "3. Add your key to the SSH agent:"
    info "   ssh-add ~/.ssh/id_ed25519"
    echo
    info "4. Test the connection manually:"
    info "   ssh -T git@hf.co"
    echo
    
    # Check if huggingface-cli is available as a fallback
    if command -v huggingface-cli &> /dev/null; then
        log "Checking HTTPS fallback..."
        if huggingface-cli whoami &> /dev/null; then
            log "✅ HTTPS authentication available as fallback"
        else
            warn "❌ HTTPS authentication not configured"
            info "Configure with: huggingface-cli login"
        fi
    else
        warn "huggingface-cli not installed"
        info "Install with: pip install huggingface_hub[cli]"
    fi
    
    exit 1
fi

log "HuggingFace SSH configuration validated! 🎉" 