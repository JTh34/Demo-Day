# 🛠️ Scripts de Développement PuppyCompanion

Ce répertoire contient tous les scripts nécessaires pour automatiser le workflow de développement, test et déploiement de PuppyCompanion.

## 📋 Scripts Disponibles

### 🐍 `setup_dev_env.sh` - Configuration Environnement Python 3.12
**Usage:** `./setup_dev_env.sh`

Configure un environnement de développement Python 3.12 spécialement optimisé pour Apple Silicon M4 et `unstructured.io`.

```bash
# Configuration initiale (une seule fois)
./setup_dev_env.sh
```

**Ce script :**
- ✅ Crée `demoday_challenge/venv_dev_312/` avec Python 3.12
- ✅ Installe `unstructured[local-inference]` compatible M4
- ✅ Configure Jupyter avec kernel dédié
- ✅ Enregistre le kernel "PuppyCompanion Dev (Python 3.12)"

---

### 🚀 `start_dev.sh` - Lancement Rapide Développement
**Usage:** `./start_dev.sh`

Lance rapidement l'environnement de développement Python 3.12 avec Jupyter Lab.

```bash
# Lancement rapide pour développement
./start_dev.sh
```

**Fonctionnalités :**
- ✅ Active automatiquement l'environnement Python 3.12
- ✅ Lance Jupyter Lab sur port 8888
- ✅ Ouvre automatiquement le navigateur
- ✅ Affiche les instructions pour le kernel

---

### 🔄 `sync_to_app.sh` - Synchronisation
**Usage:** `./sync_to_app.sh [--dry-run]`

Synchronise les modifications du répertoire de développement vers l'application de production.

```bash
# Voir ce qui sera synchronisé sans faire de modifications
./sync_to_app.sh --dry-run

# Synchroniser effectivement
./sync_to_app.sh
```

**Fichiers synchronisés :**
- `agent_workflow.py`
- `rag_system.py`
- `embedding_models.py`
- `document_loader.py`
- `evaluation.py`
- `data/` (répertoire complet)
- `pyproject.toml` (si différent)

**Exclusions automatiques :**
- `venv_dev_312/` (environnement Python 3.12)
- `__pycache__/`, `.ipynb_checkpoints/`
- `*.pyc`

---

### 🧪 `test_app.sh` - Tests Locaux
**Usage:** `./test_app.sh [--port PORT]`

Lance l'application en local pour les tests avec vérifications automatiques.

```bash
# Test sur le port par défaut (8000)
./test_app.sh

# Test sur un port spécifique
./test_app.sh --port 8080
```

**Vérifications automatiques :**
- ✅ Environnement virtuel
- ✅ Fichier `.env` et clés API
- ✅ Données préprocessées
- ✅ Import de l'application
- ✅ Lancement de Chainlit

---

### 🔑 `test_hf_ssh.sh` - Test SSH HuggingFace
**Usage:** `./test_hf_ssh.sh`

Vérifie que votre configuration SSH pour HuggingFace fonctionne correctement.

```bash
./test_hf_ssh.sh
```

**Vérifications :**
- ✅ Présence des clés SSH
- ✅ Connexion à `git@hf.co`
- ✅ Capacité de clonage Git
- ✅ Fallback HTTPS si nécessaire

---

### 🌐 `deploy_hf.sh` - Déploiement HuggingFace
**Usage:** `./deploy_hf.sh [--space-name SPACE_NAME]`

Déploie l'application sur HuggingFace Spaces avec SSH (ou HTTPS en fallback).

```bash
# Déploiement interactif
./deploy_hf.sh

# Déploiement vers un espace spécifique
./deploy_hf.sh --space-name puppycompanion-v2
```

**Fonctionnalités :**
- 🔐 Utilise SSH par défaut (plus rapide)
- 🔄 Fallback automatique vers HTTPS
- 🛡️ Vérifications de sécurité (pas de clés API)
- 📦 Génération automatique de `requirements.txt`
- 🚀 Déploiement automatisé

---

### 🚀 `full_deploy.sh` - Déploiement Complet
**Usage:** `./full_deploy.sh [OPTIONS]`

Script tout-en-un qui automatise l'ensemble du workflow.

```bash
# Déploiement complet interactif
./full_deploy.sh

# Déploiement vers un espace spécifique
./full_deploy.sh --space-name puppycompanion-v2

# Ignorer certaines étapes
./full_deploy.sh --skip-sync --skip-test

# Voir toutes les options
./full_deploy.sh --help
```

**Options disponibles :**
- `--space-name NAME` : Nom de l'espace HuggingFace
- `--skip-sync` : Ignorer la synchronisation
- `--skip-test` : Ignorer les tests locaux
- `--help` : Afficher l'aide

**Étapes automatisées :**
1. 🔄 Synchronisation des modifications
2. 🧪 Tests locaux
3. 🔑 Vérification SSH HuggingFace
4. 🌐 Déploiement sur HuggingFace Spaces

## 🎯 Workflows Recommandés

### 🔬 Développement avec Python 3.12
```bash
# Configuration initiale (une seule fois)
./setup_dev_env.sh

# Développement quotidien
./start_dev.sh  # Lance Jupyter avec Python 3.12
# ... développer dans les notebooks avec unstructured.io ...

# Synchroniser vers production
./sync_to_app.sh --dry-run  # Vérifier
./sync_to_app.sh            # Synchroniser
```

### 🚀 Déploiement Rapide
```bash
# Déploiement complet en une commande
./full_deploy.sh --space-name mon-espace
```

### 🔧 Débogage
```bash
# Tester chaque composant individuellement
./test_hf_ssh.sh           # SSH OK ?
./sync_to_app.sh --dry-run # Quoi synchroniser ?
./test_app.sh              # App fonctionne ?
./deploy_hf.sh             # Déploiement manuel
```

## 🐍 Architecture Python Dual

### **Développement (Python 3.12)**
- **Répertoire** : `demoday_challenge/`
- **Environnement** : `venv_dev_312/`
- **Usage** : Notebooks, `unstructured.io`, expérimentation
- **Lancement** : `./start_dev.sh`

### **Production (Python 3.11)**
- **Répertoire** : `puppycompanion-app/`
- **Environnement** : `venv_puppycompanion/`
- **Usage** : Application Chainlit, déploiement HF
- **Lancement** : `./test_app.sh`

## 🛡️ Sécurité

- ❌ **Jamais de clés API dans le code**
- ✅ **Clés API uniquement dans `.env`**
- ✅ **Vérifications automatiques avant déploiement**
- ✅ **Exclusion automatique des fichiers sensibles**
- ✅ **Isolation des environnements Python**

## 🆘 Dépannage

### Problème : unstructured.io ne fonctionne pas
```bash
# Vérifier l'environnement Python 3.12
cd demoday_challenge/
source venv_dev_312/bin/activate
python --version  # Doit être 3.12.x
```

### Problème : Kernel Jupyter non trouvé
```bash
# Lister les kernels disponibles
jupyter kernelspec list

# Reconfigurer l'environnement
./setup_dev_env.sh
```

### Problème : SSH échoue
```bash
./test_hf_ssh.sh
# Suivre les instructions affichées
```

### Problème : Synchronisation inattendue
```bash
./sync_to_app.sh --dry-run
# Vérifier ce qui sera modifié
```

### Problème : Tests locaux échouent
```bash
cd puppycompanion-app/
source venv_puppycompanion/bin/activate
python -c "import app"
```

---

**💡 Conseil :** Utilisez `./start_dev.sh` pour le développement et `./full_deploy.sh` pour le déploiement !

**🎯 Architecture :** Python 3.12 (dev) → Synchronisation → Python 3.11 (prod) → HuggingFace 🚀 