# 🚀 PuppyCompanion Development Workflow

This guide describes the complete process of developing, testing, and deploying the PuppyCompanion application from development to HuggingFace Spaces.

## 📁 Project Structure

```
AIE6-demoday/
├── demoday_challenge/          # 🔬 Development & Experimentation
│   ├── main_notebook.ipynb     # Main development notebook
│   ├── embedding_finetuning.ipynb
│   ├── evaluation.py           # Metrics and evaluations
│   ├── agent_workflow.py       # Agent logic
│   ├── rag_system.py          # RAG system implementation
│   ├── embedding_models.py     # Embedding models
│   ├── document_loader.py      # Document processing
│   └── data/                   # Development data
├── puppycompanion-app/         # 🚀 Production Application
│   ├── app.py                  # Chainlit application
│   ├── Dockerfile              # Docker configuration
│   ├── pyproject.toml          # Dependencies
│   ├── README.md               # Application documentation
│   ├── .env                    # Environment variables (local only)
│   ├── venv_puppycompanion/    # Virtual environment
│   └── data/                   # Production data
├── sync_to_app.sh              # 🔄 Synchronization script
├── test_app.sh                 # 🧪 Local testing script
├── test_hf_ssh.sh              # 🔑 SSH configuration test
├── deploy_hf.sh                # 🌐 HuggingFace deployment
├── full_deploy.sh              # 🚀 Complete deployment workflow
├── WORKFLOW.md                 # 📖 This guide
└── README_SCRIPTS.md           # 📋 Scripts documentation
```

## 🔄 Development Workflow

### Phase 1: Development & Experimentation
*Working directory: `demoday_challenge/`*

This is where all development and experimentation happens:

```bash
cd demoday_challenge/

# Develop in Jupyter notebooks
jupyter lab main_notebook.ipynb

# Test new features and algorithms
python evaluation.py

# Fine-tune embedding models
jupyter lab embedding_finetuning.ipynb

# Experiment with RAG improvements
python rag_system.py
```

**Key Development Files:**
- `main_notebook.ipynb` - Main development notebook
- `agent_workflow.py` - Core agent logic
- `rag_system.py` - RAG system implementation
- `evaluation.py` - Metrics and performance evaluation
- `embedding_models.py` - Custom embedding models
- `document_loader.py` - Document processing pipeline

### Phase 2: Synchronization to Production
*Transfer changes to the production app*

```bash
# Preview what will be synchronized
./sync_to_app.sh --dry-run

# Synchronize changes to production app
./sync_to_app.sh
```

**Automatically Synchronized Files:**
- `agent_workflow.py` → `puppycompanion-app/agent_workflow.py`
- `rag_system.py` → `puppycompanion-app/rag_system.py`
- `embedding_models.py` → `puppycompanion-app/embedding_models.py`
- `document_loader.py` → `puppycompanion-app/document_loader.py`
- `evaluation.py` → `puppycompanion-app/evaluation.py`
- `data/` → `puppycompanion-app/data/` (complete directory)
- `pyproject.toml` → `puppycompanion-app/pyproject.toml` (if different)

### Phase 3: Local Testing
*Test the application locally before deployment*

```bash
# Complete local testing with automatic checks
./test_app.sh

# Test on a specific port
./test_app.sh --port 8080
```

**Automatic Verification Checks:**
- ✅ Virtual environment activation
- ✅ `.env` file and API keys presence
- ✅ Preprocessed data availability
- ✅ Application imports and dependencies
- ✅ Chainlit application startup

### Phase 4: Deployment to HuggingFace Spaces
*Deploy to production environment*

```bash
# Interactive deployment
./deploy_hf.sh

# Deploy to specific space
./deploy_hf.sh --space-name puppycompanion-v2

# Complete automated workflow
./full_deploy.sh --space-name my-space
```

**Pre-deployment Security Checks:**
- ✅ Required files present
- ✅ No API keys in source code
- ✅ HuggingFace authentication working
- ✅ Automatic `requirements.txt` generation
- ✅ SSH/HTTPS connectivity test

## 🛠️ Initial Setup

### 1. First-Time Installation

```bash
cd puppycompanion-app/

# Create virtual environment
python3 -m venv venv_puppycompanion
source venv_puppycompanion/bin/activate

# Install dependencies
pip install -e .

# Create environment file
cp .env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_openai_key
# TAVILY_API_KEY=your_tavily_key
```

### 2. HuggingFace Configuration

#### Option A: SSH (Recommended - Faster & More Secure)

```bash
# Test your existing SSH configuration
./test_hf_ssh.sh

# If test fails, configure SSH:
# 1. Generate SSH key (if not already done)
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2. Add public key to your HuggingFace profile
cat ~/.ssh/id_ed25519.pub
# Copy the output and add it at: https://huggingface.co/settings/keys

# 3. Test the connection
ssh -T git@hf.co
```

#### Option B: HTTPS (Fallback)

```bash
# Install HuggingFace CLI
pip install huggingface_hub[cli]

# Authenticate
huggingface-cli login
```

## 🚀 Automated Workflows

### Quick Development Cycle
```bash
# 1. Develop in demoday_challenge/
cd demoday_challenge/
# ... make changes ...

# 2. Sync and test
cd ../
./sync_to_app.sh --dry-run  # Preview changes
./sync_to_app.sh            # Apply changes
./test_app.sh               # Test locally
```

### Complete Deployment Pipeline
```bash
# One-command deployment with all steps
./full_deploy.sh --space-name my-puppycompanion

# Skip certain steps if needed
./full_deploy.sh --skip-sync --skip-test --space-name my-space
```

### Available Full Deploy Options
- `--space-name NAME` - Specify HuggingFace space name
- `--skip-sync` - Skip synchronization step
- `--skip-test` - Skip local testing step
- `--help` - Show all available options

## 📊 Evaluation & Metrics

The evaluation system is developed in `demoday_challenge/evaluation.py` and includes:

- **RAG Accuracy**: Quality of document retrieval
- **Response Consistency**: LLM-as-judge evaluation
- **Response Time**: Performance benchmarking
- **Topic Coverage**: Thematic analysis
- **User Satisfaction**: Feedback metrics

## 🔧 Troubleshooting Guide

### SSH Connection Issues
```bash
# Test SSH configuration
./test_hf_ssh.sh

# If it fails, check:
# 1. SSH key exists: ls ~/.ssh/
# 2. Key added to HF: https://huggingface.co/settings/keys
# 3. SSH agent: ssh-add ~/.ssh/id_ed25519
```

### Synchronization Problems
```bash
# Check what will be synchronized
./sync_to_app.sh --dry-run

# Verify source files exist
ls -la demoday_challenge/
```

### Local Testing Failures
```bash
# Manual testing steps
cd puppycompanion-app/
source venv_puppycompanion/bin/activate
python -c "import app"
chainlit run app.py
```

### Deployment Issues
```bash
# Check HuggingFace authentication
huggingface-cli whoami

# Verify required files
ls -la puppycompanion-app/

# Test SSH connectivity
./test_hf_ssh.sh
```

## 📋 Best Practices

### ✅ Do's
- Always use `--dry-run` before synchronizing
- Test locally before every deployment
- Keep API keys in `.env` files only
- Document new features and changes
- Use the development directory for experimentation
- Run the complete workflow for important releases

### ❌ Don'ts
- Never edit files directly in `puppycompanion-app/`
- Never commit API keys to version control
- Don't deploy without local testing
- Don't skip the dry-run synchronization check
- Avoid manual file copying between directories

## 🔐 Security Considerations

- **API Keys**: Always stored in `.env` files, never in source code
- **Automatic Scanning**: Scripts check for API keys before deployment
- **File Exclusions**: Sensitive files automatically excluded from deployment
- **SSH Authentication**: Preferred over HTTPS for better security
- **Environment Isolation**: Separate development and production environments

## 📈 Performance Optimization

- **SSH Deployment**: Faster than HTTPS for large files
- **Incremental Sync**: Only changed files are synchronized
- **Local Testing**: Catch issues before expensive cloud deployment
- **Automated Checks**: Reduce manual verification time
- **Background Processing**: Long-running deployments don't block terminal

---

**💡 Pro Tip:** Start with `./full_deploy.sh --help` to see all available options and customize your workflow!

**🎯 Quick Start:** For first-time users, run `./test_hf_ssh.sh` to verify your setup, then use `./full_deploy.sh` for a guided deployment experience.