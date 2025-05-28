# 🚀 Guide de Déploiement PuppyCompanion

Ce document explique la marche à suivre complète pour déployer PuppyCompanion sur HuggingFace Spaces en utilisant SSH.

## 📋 Prérequis

- ✅ Configuration SSH HuggingFace fonctionnelle
- ✅ Application PuppyCompanion dans `puppycompanion-app/`
- ✅ Scripts de déploiement dans le répertoire racine

## 🎯 Méthode Recommandée : Création Manuelle + SSH

### Étape 1 : Créer l'Espace HuggingFace Manuellement

1. **Aller sur HuggingFace** : https://huggingface.co/new-space

2. **Configurer l'espace** :
   - **Space name** : `puppycompanion-v2` (ou le nom de votre choix)
   - **License** : Apache 2.0 (recommandé)
   - **SDK** : `Docker` ⚠️ **IMPORTANT**
   - **Visibility** : `Public` ou `Private` selon vos besoins
   - **Hardware** : CPU basic (gratuit) ou GPU selon vos besoins

3. **Cliquer sur "Create Space"**

4. **Noter l'URL** : `https://huggingface.co/spaces/JTh34/puppycompanion-v2`

### Étape 2 : Vérifier la Configuration SSH

```bash
# Tester votre configuration SSH
./test_hf_ssh.sh
```

**Résultat attendu :**
```
[SSH-TEST] ✅ Connexion SSH à HuggingFace réussie!
[SSH-TEST] ✅ Clonage Git SSH fonctionnel!
[SSH-TEST] Configuration SSH HuggingFace validée! 🎉
```

### Étape 3 : Déployer l'Application

```bash
# Depuis le répertoire racine AIE6-demoday/
./deploy_hf.sh --space-name puppycompanion-v2
```

## 🔄 Workflow Git Automatique

Le script `deploy_hf.sh` gère automatiquement tout le processus Git :

### 1. **Clonage SSH Automatique**
```bash
git clone "git@hf.co:spaces/JTh34/puppycompanion-v2" /tmp/hf_deploy_$$
```

### 2. **Synchronisation des Fichiers**
```bash
rsync -av --exclude='.git' --exclude='venv_*' --exclude='.env' \
    --exclude='__pycache__' --exclude='.chainlit' ./ /tmp/hf_deploy_$$/
```

### 3. **Commit Automatique**
```bash
git add .
git commit -m "Deploy PuppyCompanion $(date '+%Y-%m-%d %H:%M:%S')"
```

### 4. **Push SSH vers HuggingFace**
```bash
git push  # Via SSH - rapide et sécurisé
```

## 📁 Fichiers Déployés

### ✅ **Fichiers Inclus :**
- `app.py` - Application Chainlit
- `Dockerfile` - Configuration Docker
- `README.md` - Documentation
- `requirements.txt` - Dépendances (généré automatiquement)
- `agent_workflow.py` - Logique de l'agent
- `rag_system.py` - Système RAG
- `embedding_models.py` - Modèles d'embedding
- `document_loader.py` - Chargement de documents
- `evaluation.py` - Métriques
- `data/` - Données préprocessées

### ❌ **Fichiers Exclus (Sécurité) :**
- `.env` - Variables d'environnement locales
- `venv_*` - Environnements virtuels
- `__pycache__` - Cache Python
- `.chainlit` - Configuration locale Chainlit
- `.git` - Historique Git local

## 🛡️ Vérifications de Sécurité Automatiques

Le script effectue automatiquement ces vérifications :

1. **Fichiers requis présents** :
   - `app.py`, `Dockerfile`, `README.md`, `data/preprocessed_chunks.json`

2. **Pas de clés API dans le code** :
   - Scan automatique des patterns `sk-` dans les fichiers

3. **Génération automatique de requirements.txt** :
   - Depuis `pyproject.toml` si nécessaire

4. **Test de connectivité SSH** :
   - Vérification de `git@hf.co` avant déploiement

## 🚀 Déploiement Complet Automatisé

Pour un workflow complet de développement à déploiement :

```bash
# Workflow complet en une commande
./full_deploy.sh --space-name puppycompanion-v2

# Ou étape par étape :
./sync_to_app.sh --dry-run    # Vérifier les changements
./sync_to_app.sh              # Synchroniser
./test_app.sh                 # Tester localement
./deploy_hf.sh --space-name puppycompanion-v2  # Déployer
```

## 📊 Suivi du Déploiement

### **Pendant le Déploiement :**
```
[DEPLOY] Préparation du déploiement pour l'espace: JTh34/puppycompanion-v2
[DEPLOY] Vérifications pré-déploiement...
[DEPLOY] Génération de requirements.txt...
[DEPLOY] Vérification de la connexion SSH HuggingFace...
[DEPLOY] Connexion SSH HuggingFace OK
[DEPLOY] Clonage du repository...
[DEPLOY] Copie des fichiers...
[DEPLOY] Commit des modifications...
[DEPLOY] Déploiement vers HuggingFace...
[DEPLOY] Déploiement terminé avec succès!
```

### **Après le Déploiement :**
1. **URL de l'application** : https://huggingface.co/spaces/JTh34/puppycompanion-v2
2. **Temps de build** : 2-5 minutes (Docker)
3. **Logs disponibles** : Onglet "Logs" sur HuggingFace

## 🔧 Dépannage

### **Problème : SSH échoue**
```bash
./test_hf_ssh.sh
# Suivre les instructions affichées
```

### **Problème : Espace n'existe pas**
```
[ERROR] Échec du clonage SSH. Vérifiez le nom d'utilisateur et l'espace.
```
**Solution :** Vérifier que l'espace `JTh34/puppycompanion-v2` existe sur HuggingFace

### **Problème : Fichiers manquants**
```
[ERROR] Fichier requis manquant: app.py
```
**Solution :** Exécuter `./sync_to_app.sh` depuis `demoday_challenge/`

### **Problème : Clés API détectées**
```
[ERROR] Clés API détectées dans .env - ne les incluez pas dans le déploiement!
```
**Solution :** Les clés API doivent être configurées dans les "Settings" de l'espace HuggingFace

## 🎯 Bonnes Pratiques

### ✅ **À Faire :**
- Toujours tester avec `./test_hf_ssh.sh` avant le premier déploiement
- Utiliser `--dry-run` pour vérifier les synchronisations
- Créer l'espace manuellement une seule fois
- Garder les clés API dans les variables d'environnement HuggingFace

### ❌ **À Éviter :**
- Ne jamais inclure de clés API dans le code source
- Ne pas modifier directement les fichiers sur HuggingFace
- Ne pas oublier de synchroniser depuis `demoday_challenge/`

## 📈 Workflow de Développement Recommandé

```
1. Développer dans demoday_challenge/ 🔬
   ↓
2. ./sync_to_app.sh --dry-run 👀
   ↓
3. ./sync_to_app.sh 🔄
   ↓
4. ./test_app.sh 🧪
   ↓
5. ./deploy_hf.sh --space-name puppycompanion-v2 🚀
   ↓
6. Vérifier sur https://huggingface.co/spaces/JTh34/puppycompanion-v2 ✅
```

---

**💡 Conseil :** Une fois l'espace créé manuellement, tous les futurs déploiements sont entièrement automatisés avec SSH !

**🎯 Résumé :** Création manuelle (1 fois) + SSH (toujours) = Workflow optimal ! 🚀 