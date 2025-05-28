# 🐍 Guide de Gestion des Versions Python - PuppyCompanion

Ce guide explique comment gérer l'architecture à deux environnements Python pour PuppyCompanion sur Apple Silicon M4.

## 🎯 Problématique

- **Développement** : Python 3.12 requis pour `unstructured.io` sur Apple Silicon M4
- **Production** : Python 3.11 pour compatibilité HuggingFace Spaces
- **Solution** : Architecture à deux environnements séparés

## 📁 Architecture des Environnements

```
AIE6-demoday/
├── demoday_challenge/           # 🔬 DÉVELOPPEMENT
│   ├── venv_dev_312/           # Python 3.12 pour notebooks
│   ├── main_notebook.ipynb     # Notebooks avec unstructured.io
│   └── ...
├── puppycompanion-app/         # 🚀 PRODUCTION
│   ├── venv_puppycompanion/    # Python 3.11 pour déploiement
│   ├── app.py                  # Application Chainlit
│   └── ...
└── setup_dev_env.sh           # Script de configuration
```

## 🛠️ Configuration Initiale

### Étape 1 : Installer Python 3.12

```bash
# Via Homebrew (recommandé)
brew install python@3.12

# Via pyenv (alternative)
pyenv install 3.12.0
pyenv local 3.12.0  # Dans demoday_challenge/
```

### Étape 2 : Configurer l'environnement de développement

```bash
# Lancer le script de configuration
./setup_dev_env.sh
```

**Ce script va :**
- ✅ Créer `demoday_challenge/venv_dev_312/` avec Python 3.12
- ✅ Installer toutes les dépendances incluant `unstructured.io`
- ✅ Configurer Jupyter avec un kernel dédié
- ✅ Enregistrer le kernel "PuppyCompanion Dev (Python 3.12)"

## 🔄 Workflow de Développement

### 1. **Développement dans les Notebooks**

```bash
# Aller dans le répertoire de développement
cd demoday_challenge/

# Activer l'environnement Python 3.12
source venv_dev_312/bin/activate

# Lancer Jupyter Lab
jupyter lab
```

**Dans Jupyter :**
- Sélectionner le kernel : **"PuppyCompanion Dev (Python 3.12)"**
- Utiliser `unstructured.io` sans problème
- Développer et expérimenter librement

### 2. **Synchronisation vers Production**

```bash
# Retourner au répertoire racine
cd ../

# Synchroniser les modifications (sans les environnements virtuels)
./sync_to_app.sh --dry-run  # Vérifier
./sync_to_app.sh            # Synchroniser
```

### 3. **Tests en Production**

```bash
# Tester avec l'environnement Python 3.11
./test_app.sh

# Déployer si tout fonctionne
./deploy_hf.sh --space-name puppycompanion-v2
```

## 🔧 Gestion des Dépendances

### **Développement (Python 3.12)**
- `unstructured[local-inference]>=0.11.0` ✅ Compatible M4
- Toutes les bibliothèques de recherche et expérimentation
- Jupyter, IPython, outils de développement

### **Production (Python 3.11)**
- Dépendances minimales pour Chainlit
- Compatibilité HuggingFace Spaces
- Pas d'`unstructured.io` (données déjà préprocessées)

## 📊 Kernels Jupyter Disponibles

Après configuration, vous aurez :

1. **"PuppyCompanion Dev (Python 3.12)"** 
   - Pour développement avec `unstructured.io`
   - Localisation : `demoday_challenge/venv_dev_312/`

2. **"Python 3"** (système)
   - Kernel par défaut du système

3. **"venv_puppycompanion"** (optionnel)
   - Si vous installez ipykernel dans l'env de production

## 🚨 Points d'Attention

### ✅ **À Faire :**
- Toujours utiliser le bon environnement pour chaque tâche
- Développer dans `demoday_challenge/` avec Python 3.12
- Tester en production avec Python 3.11
- Synchroniser régulièrement entre les environnements

### ❌ **À Éviter :**
- Mélanger les environnements virtuels
- Installer `unstructured.io` dans l'environnement de production
- Oublier de synchroniser après développement
- Déployer sans tester en Python 3.11

## 🔄 Commandes Rapides

### **Développement (Python 3.12)**
```bash
cd demoday_challenge/
source venv_dev_312/bin/activate
jupyter lab  # Kernel: "PuppyCompanion Dev (Python 3.12)"
```

### **Production (Python 3.11)**
```bash
cd puppycompanion-app/
source venv_puppycompanion/bin/activate
chainlit run app.py
```

### **Synchronisation**
```bash
# Depuis la racine
./sync_to_app.sh --dry-run
./sync_to_app.sh
```

## 🐛 Dépannage

### **Problème : unstructured.io ne fonctionne pas**
```bash
# Vérifier la version Python
python --version  # Doit être 3.12.x

# Vérifier l'architecture
uname -m  # Doit être arm64

# Réinstaller si nécessaire
pip uninstall unstructured
pip install "unstructured[local-inference]>=0.11.0"
```

### **Problème : Kernel Jupyter non trouvé**
```bash
# Lister les kernels disponibles
jupyter kernelspec list

# Réenregistrer le kernel
cd demoday_challenge/
source venv_dev_312/bin/activate
python -m ipykernel install --user --name="puppycompanion-dev-312" --display-name="PuppyCompanion Dev (Python 3.12)"
```

### **Problème : Synchronisation échoue**
```bash
# Vérifier les différences
./sync_to_app.sh --dry-run

# Vérifier les permissions
ls -la demoday_challenge/
ls -la puppycompanion-app/
```

## 📈 Avantages de cette Architecture

- ✅ **Développement optimal** : Python 3.12 + unstructured.io sur M4
- ✅ **Production stable** : Python 3.11 compatible HuggingFace
- ✅ **Isolation complète** : Pas de conflits entre environnements
- ✅ **Workflow fluide** : Synchronisation automatisée
- ✅ **Flexibilité** : Chaque environnement optimisé pour son usage

---

**💡 Conseil :** Cette architecture vous permet d'exploiter pleinement les capacités de votre Mac M4 en développement tout en gardant une compatibilité maximale en production !

**🎯 Résumé :** Développement Python 3.12 → Synchronisation → Production Python 3.11 → Déploiement HuggingFace 🚀 