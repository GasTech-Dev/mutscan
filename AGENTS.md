# Repository Guidelines

## Project Layout
- `Start/` contient le script historique `All.py` et le jeu de données par défaut `sequences.fasta`.
- `Mutation/` regroupe `simulate_mutations.py` et les futurs outils d’analyse de variations.
- Les tests résident dans `tests/`; ajoutez des fixtures si vous introduisez des FASTA spécifiques à un scénario.
- `pyproject.toml` et `uv.lock` fixent Python 3.13, Biopython et Pytest; exécutez `uv sync` après chaque changement de dépendances.

## Commandes essentielles
- `uv sync` prépare l’environnement virtuel.
- `uv run python Start/All.py` calcule les contenus GC de `Start/sequences.fasta`.
- `uv run python Mutation/simulate_mutations.py --seed 42 --mutations 3 --samples 5` lance cinq séries de mutations reproductibles.
- `uv run python Mutation/simulate_mutations.py --mutations-csv results/mutations_summary.csv …` exporte un CSV prêt pour l’analyse MVP.
- `uv run python -m pytest` exécute l’ensemble des tests unitaires.
- `uv run python Mutation/analyze_mutations.py results/mutations_summary.csv --fasta Start/sequences.fasta --critical-csv results/mutations_critiques.csv --critical-fasta results/mutations_critiques.fasta --context-window 12` génère les tableaux, figures, scores de risque et exporte les mutations critiques avec leur contexte protéique.

## Paramètres de simulation
- `--fasta CHEMIN` remplace le dataset, ex. `--fasta data/sars2.fasta`.
- `--record-id IDENTIFIANT` cible une entrée spécifique d’un FASTA multi-séquences.
- `--mutations N` force N substitutions ponctuelles par simulation; `--samples N` répète l’expérience.
- `--seed N` fige le tirage aléatoire pour comparer les effets entre branches.
- `--preview-aa N` affiche les N premiers acides aminés de chaque protéine (mettre `0` pour éviter l’aperçu).
- `--protein-fasta chemin/proteines.fasta` exporte la protéine de référence et chaque protéine mutée pour analyse externe.
- `--mutations-csv chemin/mutations.csv` enregistre les mutations acceptées avec positions aa/nt, scores BLOSUM/Grantham et probabilité d’acceptation pour l’analyse post-run.
- `--annotation annotations.csv` charge une table `start,end,name[,product]` pour nommer les protéines atteintes (défaut: détection SARS-CoV-2).
- `--transition-weight W` et `--transversion-weight W` ajustent le biais Ts/Tv des substitutions.
- `--hotspot-cpg-weight W` multiplie la probabilité de tirage pour les positions CpG reconnues.
- `--feature-weight NOM=M` applique un facteur par protéine ou produit (option répétable).
- `--selection-strength S` module l'influence de BLOSUM62 sur l'acceptation des mutations.
- `--codon-usage-weight W` renforce ou atténue la préférence d'usage des codons observés.
- `--indel-probability P` tente un indel in-frame (rare) par simulation.
- `--indel-max-codons N` borne la longueur des insertions/suppressions conservées.
- `--indel-proofreading P` fixe la probabilité qu'un indel échappe au proofreading.

Chaque simulation reporte désormais le ratio `dN/dS` et les indels acceptés ou rejetés pour faciliter le suivi des pressions de sélection.
Un récapitulatif global conclut chaque batch avec le tableau de sévérité (score, BLOSUM, probabilité d'acceptation) et le bilan par protéine.
Les sorties détaillent aussi les distances de Grantham, la variation d’hydrophobicité et les drapeaux critiques (stop, Pro/Cys/Gly pertes) pour prioriser les validations. L’analyse peut en plus agréger les mutations `critique` dans un CSV et un FASTA contextuels pour BLAST/AlphaFold (options `--critical-csv`, `--critical-fasta`, `--context-window`).

## Style de code
- Respectez PEP 8 : indentations de quatre espaces, `snake_case` pour fonctions/variables, `PascalCase` réservé aux classes.
- Utilisez les majuscules uniquement pour les constantes et documentez brièvement les hypothèses biologiques dans les docstrings.

## Tests & qualité
- Placez les tests dans `tests/test_<module>.py` et vérifiez à la fois les mutations silencieuses et non silencieuses.
- Couvrir les cas de bases ambiguës, de trim final (`prepare_coding_sequence`) et les effets protéiques attendus.
- Lancez `uv run python -m pytest` avant chaque PR et fournissez les échantillons FASTA nécessaires.

## Contributions
- Rédigez des commits à l’impératif (`Document mutation options`) et limitez les corps à 72 caractères par ligne.
- Les PRs recensent les commandes exécutées, les jeux de données utilisés et l’impact observé (ex. capture d’écran ou diff d’output).
- Stockez les datasets volumineux ou sensibles hors dépôt et mentionnez leur source ou procédure d’accès.

## TODO évolutions simulation
- Calibrer les paramètres de sélection (strength, proofreading) sur des jeux de données empiriques.
- Ajouter un rapport global multi-échantillons consolidant dN/dS et distribution des indels.
- Étendre la simulation aux mutations par glissement (indels hors frame) avec gestion des codons stop.
- Résoudre demain l'erreur `ModuleNotFoundError: No module named 'Mutation'` observée en lançant `python Mutation/generate_heatmap.py`.
