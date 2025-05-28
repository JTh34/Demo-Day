# PuppyCompanion - Projet AIE6 Demo Day

## Description
Application Chainlit pour l'assistance aux propriétaires de chiens, développée dans le cadre du Demo Day AIE6.

## Structure du Projet

```
AIE6-demoday/
├── demoday_challenge/     # Répertoire de développement et expérimentation
├── puppycompanion-app/    # Application de production pour HuggingFace Spaces
├── scripts/               # Scripts d'automatisation du workflow
│   ├── sync_to_app.sh    # Synchronisation dev → prod
│   ├── test_app.sh       # Tests locaux
│   ├── deploy_hf.sh      # Déploiement HuggingFace
│   ├── full_deploy.sh    # Workflow complet
│   └── git_workflow.sh   # Gestion Git automatisée
└── docs/                  # Documentation
    ├── WORKFLOW.md
    ├── DEPLOYMENT_GUIDE.md
    ├── README_SCRIPTS.md
    └── GIT_GUIDE.md       # Guide de versioning
```

## Workflow de Développement

1. **Développement** : Travaillez dans `demoday_challenge/`
2. **Sauvegarde Git** : `./git_workflow.sh backup` pour sauvegarder régulièrement
3. **Synchronisation** : `./sync_to_app.sh` pour copier vers `puppycompanion-app/`
4. **Tests locaux** : `./test_app.sh` pour vérifier l'application
5. **Commit final** : `./git_workflow.sh commit` avec message descriptif
6. **Déploiement** : `./deploy_hf.sh` pour publier sur HuggingFace Spaces

## Installation et Configuration

### Prérequis
- Python 3.8+
- Git configuré avec votre identité
- Compte HuggingFace avec SSH configuré
- Clés API (OpenAI, Qdrant)

### Configuration initiale
```bash
# Configurer Git (première fois)
git config user.name "Votre Nom"
git config user.email "votre.email@example.com"

# Créer l'environnement virtuel
python -m venv venv_puppycompanion
source venv_puppycompanion/bin/activate

# Installer les dépendances
cd puppycompanion-app
pip install -e .

# Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos clés API
```

## Utilisation

### Développement local
```bash
./test_app.sh
```

### Gestion Git
```bash
./git_workflow.sh status    # Voir l'état du projet
./git_workflow.sh backup    # Sauvegarde rapide
./git_workflow.sh commit    # Commit avec message
./git_workflow.sh help      # Voir toutes les options
```

### Déploiement complet
```bash
./full_deploy.sh
```

### Synchronisation uniquement
```bash
./sync_to_app.sh --dry-run  # Aperçu des changements
./sync_to_app.sh            # Synchronisation effective
```

## Suivi de Version

Ce projet utilise Git pour le suivi de version avec :
- **Dépôt local** : Historique complet des modifications
- **Script automatisé** : `git_workflow.sh` pour simplifier les opérations Git
- **Bonnes pratiques** : Messages de commit structurés, branches par fonctionnalité
- **Intégration workflow** : Git s'intègre parfaitement avec vos scripts existants

Consultez le [Guide Git](GIT_GUIDE.md) pour les détails complets.

## Technologies Utilisées
- **Chainlit** : Interface utilisateur conversationnelle
- **OpenAI** : Modèles de langage
- **Qdrant** : Base de données vectorielle
- **LangChain** : Framework pour applications LLM
- **Git** : Suivi de version et collaboration

## Documentation
- [Guide de Workflow](WORKFLOW.md)
- [Guide de Déploiement](DEPLOYMENT_GUIDE.md)
- [Documentation des Scripts](README_SCRIPTS.md)
- [Guide Git et Versioning](GIT_GUIDE.md)

## Contribution
1. Créer une branche pour votre fonctionnalité : `git checkout -b feature/nom-fonctionnalite`
2. Développer dans `demoday_challenge/`
3. Sauvegarder régulièrement : `./git_workflow.sh backup`
4. Tester avec `./test_app.sh`
5. Synchroniser avec `./sync_to_app.sh`
6. Commit final : `./git_workflow.sh commit`
7. Créer une pull request

## Licence
[Spécifier la licence du projet] 