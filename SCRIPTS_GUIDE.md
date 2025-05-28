# Guide des Scripts - PuppyCompanion

## 📋 Scripts Essentiels

Voici les scripts utiles pour votre workflow de développement, classés par usage.

---

## 🚀 **Scripts de Développement**

### 1. `start_dev.sh`
**Usage** : Configuration et démarrage de l'environnement de développement  
**Répertoire** : `/Users/jthomazo/Archives/01_Projets/02_AIM/APP/AIE6-demoday`  
**Commande** : `./start_dev.sh`

**Fonctions** :
- Crée l'environnement virtuel si nécessaire
- Active l'environnement
- Installe les dépendances
- Lance l'application en mode développement

**Quand l'utiliser** : Au début de chaque session de développement

---

## 🔄 **Scripts de Synchronisation**

### 2. `sync_to_app.sh`
**Usage** : Synchronise le code de développement vers la version production  
**Répertoire** : `/Users/jthomazo/Archives/01_Projets/02_AIM/APP/AIE6-demoday`  
**Commande** : `./sync_to_app.sh [--dry-run]`

**Options** :
- `--dry-run` : Aperçu des changements sans les appliquer
- Sans option : Synchronisation effective

**Fonctions** :
- Copie les fichiers de `demoday_challenge/` vers `puppycompanion-app/`
- Exclut automatiquement les fichiers sensibles
- Vérifie les conflits potentiels

**Quand l'utiliser** : Après avoir développé une fonctionnalité dans `demoday_challenge/`

---

## 🧪 **Scripts de Test**

### 3. `test_app.sh`
**Usage** : Test local de l'application de production  
**Répertoire** : `/Users/jthomazo/Archives/01_Projets/02_AIM/APP/AIE6-demoday`  
**Commande** : `./test_app.sh`

**Fonctions** :
- Vérifie l'environnement de production
- Lance l'application en mode test
- Vérifie les dépendances
- Teste la configuration

**Quand l'utiliser** : Avant le déploiement pour vérifier que tout fonctionne

### 4. `test_hf_ssh.sh`
**Usage** : Test de la configuration SSH pour HuggingFace  
**Répertoire** : `/Users/jthomazo/Archives/01_Projets/02_AIM/APP/AIE6-demoday`  
**Commande** : `./test_hf_ssh.sh`

**Fonctions** :
- Vérifie la connexion SSH à HuggingFace
- Teste l'authentification
- Valide la configuration Git

**Quand l'utiliser** : Pour diagnostiquer les problèmes de déploiement

---

## 🚀 **Scripts de Déploiement**

### 5. `deploy_hf.sh`
**Usage** : Déploiement sur HuggingFace Spaces  
**Répertoire** : `/Users/jthomazo/Archives/01_Projets/02_AIM/APP/AIE6-demoday`  
**Commande** : `./deploy_hf.sh`

**Fonctions** :
- Déploie l'application sur HuggingFace Spaces
- Utilise SSH (plus rapide) avec fallback HTTPS
- Vérifie les fichiers sensibles
- Génère automatiquement `requirements.txt`

**Quand l'utiliser** : Pour publier votre application en ligne

### 6. `full_deploy.sh`
**Usage** : Workflow complet de déploiement  
**Répertoire** : `/Users/jthomazo/Archives/01_Projets/02_AIM/APP/AIE6-demoday`  
**Commande** : `./full_deploy.sh [options]`

**Options** :
- `--skip-sync` : Ignore la synchronisation
- `--skip-test` : Ignore les tests
- `--force` : Force le déploiement

**Fonctions** :
- Synchronise automatiquement (`sync_to_app.sh`)
- Teste l'application (`test_app.sh`)
- Déploie sur HuggingFace (`deploy_hf.sh`)

**Quand l'utiliser** : Pour un déploiement complet automatisé

---

## 📝 **Scripts Git**

### 7. `git_workflow.sh`
**Usage** : Gestion Git simplifiée  
**Répertoire** : `/Users/jthomazo/Archives/01_Projets/02_AIM/APP/AIE6-demoday`  
**Commande** : `./git_workflow.sh [option]`

**Options principales** :
- `status` : Afficher l'état Git
- `backup` : Sauvegarde rapide avec timestamp
- `commit` : Commit interactif
- `push` : Pousser vers GitHub
- `help` : Aide complète

**Quand l'utiliser** : Pour toutes les opérations Git quotidiennes

---

## 🎯 **Workflow Recommandé**

### Développement d'une nouvelle fonctionnalité :

```bash
# 1. Démarrer l'environnement de développement
./start_dev.sh

# 2. Développer dans demoday_challenge/
# ... modifications ...

# 3. Sauvegarder régulièrement
./git_workflow.sh backup

# 4. Synchroniser vers production
./sync_to_app.sh --dry-run  # Vérifier
./sync_to_app.sh            # Appliquer

# 5. Tester
./test_app.sh

# 6. Commit final
./git_workflow.sh commit

# 7. Déployer
./deploy_hf.sh
# OU workflow complet :
./full_deploy.sh
```

### Synchronisation GitHub :

```bash
# Pousser vers GitHub (code source uniquement)
./git_workflow.sh push
# OU directement :
git push github main
```

---

## 📁 **Structure des Répertoires**

```
AIE6-demoday/                    ← Exécuter tous les scripts depuis ici
├── demoday_challenge/           ← Répertoire de développement
├── puppycompanion-app/          ← Application de production
├── start_dev.sh                 ← Développement
├── sync_to_app.sh              ← Synchronisation
├── test_app.sh                 ← Tests
├── test_hf_ssh.sh              ← Diagnostic SSH
├── deploy_hf.sh                ← Déploiement HuggingFace
├── full_deploy.sh              ← Workflow complet
└── git_workflow.sh             ← Gestion Git
```

---

## ⚠️ **Notes Importantes**

1. **Tous les scripts doivent être exécutés depuis le répertoire racine** : `/Users/jthomazo/Archives/01_Projets/02_AIM/APP/AIE6-demoday`

2. **Permissions** : Si un script n'est pas exécutable :
   ```bash
   chmod +x nom_du_script.sh
   ```

3. **Variables d'environnement** : Assurez-vous que vos clés API sont configurées dans les fichiers `.env`

4. **GitHub vs HuggingFace** :
   - GitHub : Code source uniquement (public)
   - HuggingFace : Application complète (privé)

---

## 🆘 **Dépannage Rapide**

| Problème | Script à utiliser |
|----------|-------------------|
| L'app ne démarre pas | `./start_dev.sh` |
| Erreur de synchronisation | `./sync_to_app.sh --dry-run` |
| Problème de déploiement | `./test_hf_ssh.sh` |
| Conflit Git | `./git_workflow.sh status` |
| Test avant déploiement | `./test_app.sh` |

---

## 📚 **Documentation Complète**

- [Guide Git](GIT_GUIDE.md) : Utilisation avancée de Git
- [Guide de Déploiement](DEPLOYMENT_GUIDE.md) : Déploiement détaillé
- [Workflow Complet](WORKFLOW.md) : Processus de développement complet 