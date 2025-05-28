#!/bin/bash

# Local test script for PuppyCompanion
# Usage: ./test_app.sh [--port PORT]

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[TEST]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default configuration
PORT=8000
APP_DIR="./puppycompanion-app"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check that the directory exists
if [ ! -d "$APP_DIR" ]; then
    error "Application directory not found: $APP_DIR"
    exit 1
fi

cd "$APP_DIR"

log "Checking environment..."

# Check virtual environment
if [ ! -d "venv_puppycompanion" ]; then
    error "Virtual environment not found. Please run the installation first."
    exit 1
fi

# Check .env file
if [ ! -f ".env" ]; then
    error ".env file not found. Please create it with your API keys."
    exit 1
fi

# Check API keys
if grep -q "your_openai_api_key_here" .env; then
    warn "OpenAI key not configured in .env"
    warn "Edit the .env file with your real API key"
fi

# Check data
if [ ! -f "data/preprocessed_chunks.json" ]; then
    error "Preprocessed data not found: data/preprocessed_chunks.json"
    exit 1
fi

log "Testing application import..."

# Activate virtual environment and test import
source venv_puppycompanion/bin/activate

# Import test
if ! python -c "import app; print('✓ Import successful')" 2>/dev/null; then
    error "Application import failed"
    exit 1
fi

log "Launching the application on port $PORT..."
log "The application will be available at: http://localhost:$PORT"
log "Press Ctrl+C to stop"

# Launch the application
chainlit run app.py --port "$PORT" --host 0.0.0.0 