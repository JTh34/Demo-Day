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
│   └── full_deploy.sh    # Workflow complet
└── docs/                  # Documentation
    ├── WORKFLOW.md
    ├── DEPLOYMENT_GUIDE.md
    └── README_SCRIPTS.md
```

## Workflow de Développement

1. **Développement** : Travaillez dans `demoday_challenge/`
2. **Synchronisation** : `./sync_to_app.sh` pour copier vers `puppycompanion-app/`
3. **Tests locaux** : `./test_app.sh` pour vérifier l'application
4. **Déploiement** : `./deploy_hf.sh` pour publier sur HuggingFace Spaces

## Installation et Configuration

### Prérequis
- Python 3.8+
- Compte HuggingFace avec SSH configuré
- Clés API (OpenAI, Qdrant)

### Configuration initiale
```bash
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

### Déploiement complet
```bash
./full_deploy.sh
```

### Synchronisation uniquement
```bash
./sync_to_app.sh --dry-run  # Aperçu des changements
./sync_to_app.sh            # Synchronisation effective
```

## Technologies Utilisées
- **Chainlit** : Interface utilisateur conversationnelle
- **OpenAI** : Modèles de langage
- **Qdrant** : Base de données vectorielle
- **LangChain** : Framework pour applications LLM

## Documentation
- [Guide de Workflow](WORKFLOW.md)
- [Guide de Déploiement](DEPLOYMENT_GUIDE.md)
- [Documentation des Scripts](README_SCRIPTS.md)

## Contribution
1. Créer une branche pour votre fonctionnalité
2. Développer dans `demoday_challenge/`
3. Tester avec `./test_app.sh`
4. Synchroniser avec `./sync_to_app.sh`
5. Créer une pull request

## Licence
[Spécifier la licence du projet] 