# Zero-Shot Text Classification with Small Language Models

Ce projet vise à évaluer la capacité de petits modèles de langage (comme LaMini-Flan-T5-783M) à résoudre des tâches de classification **en zero-shot**, c’est-à-dire sans entraînement sur les données cibles.

---

## 1. Objectif

Le but est de mesurer dans quelle mesure des modèles réduits de type `T5` peuvent généraliser à des tâches de classification textuelle simplement à partir d'un prompt bien formulé, sans fine-tuning.

---

## 2. Démarche adoptée

La méthodologie suit les étapes suivantes :

### a. Sélection de jeux de données de classification

Trois datasets sont utilisés :

| Dataset   | Tâche                      | Nombre de classes | Description rapide                                   |
|-----------|----------------------------|--------------------|------------------------------------------------------|
| imdb      | Sentiment Analysis         | 2                  | Classer une critique de film en positif ou négatif   |
| agnews   | ...     | 5                  | ...            |

### b. Création de fichiers de prompt

Pour chaque dataset, un fichier JSON contient :

- Un `prompt` contenant les placeholders `{TEXT}` et `{LABEL}`.
- Un dictionnaire `verbalizer` listant les labels et leurs synonymes.
- Une liste de `samples` contenant des textes labellisés.

Ces fichiers sont enregistrés dans le dossier `prompts/`.

### c. Deux méthodes de scoring

Le fichier `run_experiment.py` permet d’évaluer les modèles selon deux stratégies :

1. **Scoring par décodage** (`--scoring decoding`) :
   - Le modèle reçoit un prompt.
   - Il génère une sortie libre.
   - On compare la sortie générée aux synonymes du `verbalizer`.

2. **Scoring par label** (`--scoring label`) :
   - On génère un prompt pour chaque label possible.
   - On calcule la loss associée à chaque prompt.
   - On choisit le label avec la perte la plus faible (meilleure vraisemblance).

### d. Évaluation des performances

Les métriques suivantes sont calculées :

- Accuracy
- F1-score (avec moyenne `"binary"` pour les tâches binaires, `"weighted"` sinon)
- Baselines :
  - Prédiction de la classe majoritaire
  - Prédiction aléatoire uniforme

### e. Sauvegarde des résultats

Les résultats sont sauvegardés dans le dossier `results/` sous forme de fichiers `.json`.

---

## 3. Utilisation

### a. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### b. Création des prompts 

```bash
Les prompts sont générés depuis le fichier notebooks/analysis.ipynb
```

### c. Lancement des classifications

```bash
# Pour le sms dataset 
python run_experiment.py --dataset sms --scoring decoding --max_samples 20 --save_results # methode decoding
python run_experiment.py --dataset sms --scoring label --max_samples 20 --save_results # methode label

# Pour le imdb dataset
python run_experiment.py --dataset imdb --scoring decoding --max_samples 20 --save_results # methode decoding
python run_experiment.py --dataset imdb --scoring label --max_samples 20 --save_results # methode label

# Pour le emotion dataset
python run_experiment.py --dataset emotion --scoring decoding --max_samples 20 --save_results # methode decoding
python run_experiment.py --dataset emotion --scoring label --max_samples 20 --save_results # methode label
```

## 4. Résultats

![image](./results/summary_plot.png)
