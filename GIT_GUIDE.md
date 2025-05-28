# Guide Git pour PuppyCompanion

## Configuration Initiale

### 1. Configuration de l'identité Git
```bash
git config user.name "Votre Nom"
git config user.email "votre.email@example.com"

# Configuration globale (optionnel)
git config --global user.name "Votre Nom"
git config --global user.email "votre.email@example.com"
```

### 2. Vérification de la configuration
```bash
git config --list
```

## Workflow Git Recommandé

### Workflow Quotidien

1. **Vérifier le statut**
   ```bash
   ./git_workflow.sh status
   # ou
   git status
   ```

2. **Ajouter les modifications**
   ```bash
   ./git_workflow.sh add
   # ou
   git add .
   ```

3. **Créer un commit**
   ```bash
   ./git_workflow.sh commit
   # ou
   git commit -m "Description des changements"
   ```

4. **Sauvegarde rapide**
   ```bash
   ./git_workflow.sh backup
   ```

### Types de Commits Recommandés

#### Messages de Commit Structurés
```
feat: Nouvelle fonctionnalité
fix: Correction de bug
docs: Mise à jour documentation
style: Changements de style/formatage
refactor: Refactorisation du code
test: Ajout/modification de tests
chore: Tâches de maintenance
```

#### Exemples de Messages
```bash
git commit -m "feat: Add new RAG evaluation metrics"
git commit -m "fix: Resolve Chainlit authentication issue"
git commit -m "docs: Update deployment guide with SSH setup"
git commit -m "refactor: Simplify embedding model selection"
```

## Gestion des Branches

### Créer une Nouvelle Branche
```bash
git checkout -b feature/nouvelle-fonctionnalite
# ou
./git_workflow.sh branch
```

### Branches Recommandées
- `main` : Version stable de production
- `develop` : Branche de développement
- `feature/nom-fonctionnalite` : Nouvelles fonctionnalités
- `fix/nom-bug` : Corrections de bugs
- `hotfix/nom-correction` : Corrections urgentes

### Changer de Branche
```bash
git checkout main
git checkout develop
git checkout feature/nouvelle-fonctionnalite
```

### Fusionner une Branche
```bash
git checkout main
git merge feature/nouvelle-fonctionnalite
```

## Dépôt Distant

### Ajouter un Dépôt Distant
```bash
git remote add origin https://github.com/username/puppycompanion.git
# ou
git remote add origin git@github.com:username/puppycompanion.git
```

### Pousser vers le Dépôt Distant
```bash
git push origin main
# ou
./git_workflow.sh push
```

### Récupérer les Changements
```bash
git pull origin main
# ou
./git_workflow.sh pull
```

## Commandes Git Utiles

### Historique et Logs
```bash
git log --oneline -10          # 10 derniers commits
git log --graph --oneline      # Graphique des branches
git log --author="Nom"         # Commits d'un auteur
./git_workflow.sh log          # Via le script
```

### Annuler des Changements
```bash
git checkout -- fichier.py    # Annuler modifications d'un fichier
git reset HEAD fichier.py      # Retirer un fichier du staging
git reset --soft HEAD~1        # Annuler le dernier commit (garder les changements)
git reset --hard HEAD~1        # Annuler le dernier commit (perdre les changements)
```

### Voir les Différences
```bash
git diff                       # Changements non stagés
git diff --cached              # Changements stagés
git diff HEAD~1                # Différences avec le commit précédent
```

## Workflow Spécifique au Projet

### 1. Développement d'une Nouvelle Fonctionnalité
```bash
# 1. Créer une branche
git checkout -b feature/nouvelle-fonctionnalite

# 2. Développer dans demoday_challenge/
# ... modifications ...

# 3. Commits réguliers
./git_workflow.sh add
./git_workflow.sh commit

# 4. Synchroniser avec puppycompanion-app/
./sync_to_app.sh

# 5. Tester
./test_app.sh

# 6. Commit final
git add .
git commit -m "feat: Complete nouvelle-fonctionnalite implementation"

# 7. Fusionner dans main
git checkout main
git merge feature/nouvelle-fonctionnalite
```

### 2. Correction de Bug Urgent
```bash
# 1. Créer une branche hotfix
git checkout -b hotfix/correction-urgente

# 2. Corriger le problème
# ... modifications ...

# 3. Tester
./test_app.sh

# 4. Commit
git add .
git commit -m "fix: Resolve critical authentication issue"

# 5. Fusionner et déployer
git checkout main
git merge hotfix/correction-urgente
./deploy_hf.sh
```

### 3. Sauvegarde Régulière
```bash
# Sauvegarde automatique avec timestamp
./git_workflow.sh backup

# Ou commit manuel
git add .
git commit -m "wip: Work in progress on RAG improvements"
```

## Bonnes Pratiques

### 1. Commits Fréquents
- Commitez souvent avec des messages clairs
- Un commit = une modification logique
- Utilisez `./git_workflow.sh backup` pour les sauvegardes

### 2. Messages de Commit
- Utilisez l'impératif : "Add feature" pas "Added feature"
- Première ligne < 50 caractères
- Ligne vide puis description détaillée si nécessaire

### 3. Gestion des Fichiers
- Vérifiez le `.gitignore` régulièrement
- Ne commitez jamais les clés API ou mots de passe
- Utilisez `git status` avant chaque commit

### 4. Branches
- Une branche par fonctionnalité
- Noms de branches descriptifs
- Supprimez les branches fusionnées

## Dépannage

### Problèmes Courants

#### 1. Conflit de Fusion
```bash
git status                     # Voir les fichiers en conflit
# Éditer les fichiers pour résoudre les conflits
git add fichier-resolu.py
git commit -m "resolve: Merge conflict in fichier-resolu.py"
```

#### 2. Annuler le Dernier Commit
```bash
git reset --soft HEAD~1        # Garder les changements
git reset --hard HEAD~1        # Perdre les changements
```

#### 3. Récupérer un Fichier Supprimé
```bash
git checkout HEAD -- fichier-supprime.py
```

#### 4. Voir l'Historique d'un Fichier
```bash
git log --follow fichier.py
```

## Script d'Automatisation

Le script `git_workflow.sh` simplifie les opérations courantes :

```bash
./git_workflow.sh help         # Aide
./git_workflow.sh status       # Statut
./git_workflow.sh add          # Ajouter fichiers
./git_workflow.sh commit       # Commit interactif
./git_workflow.sh quick        # Commit rapide
./git_workflow.sh backup       # Sauvegarde
./git_workflow.sh push         # Pousser vers distant
./git_workflow.sh pull         # Récupérer du distant
./git_workflow.sh sync         # Synchronisation complète
./git_workflow.sh branch       # Gestion des branches
./git_workflow.sh log          # Historique
```

## Intégration avec le Workflow Existant

### Workflow Complet de Développement
```bash
# 1. Développement
cd demoday_challenge/
# ... modifications ...

# 2. Sauvegarde Git
./git_workflow.sh backup

# 3. Synchronisation
./sync_to_app.sh

# 4. Tests
./test_app.sh

# 5. Commit final
./git_workflow.sh add
./git_workflow.sh commit

# 6. Déploiement
./deploy_hf.sh

# 7. Push vers dépôt distant
./git_workflow.sh push
```

Ce guide vous permet de maintenir un historique propre et organisé de votre projet PuppyCompanion tout en s'intégrant parfaitement avec votre workflow de développement existant. 