# Analyse de mutations virales

## Présentation & analyses
- Simulateur `Mutation/simulate_mutations.py` : applique des mutations ponctuelles ou des indels sur une séquence codante, mesure dN/dS, scores BLOSUM62, distances de Grantham, variations d’hydrophobicité et flags critiques (stops, pertes Pro/Cys/Gly).
- Outil `Mutation/generate_heatmap.py` : agrège les logs pour produire plusieurs sorties dans `out/metrics/` :
  - `protein_accept_heatmap.png` : tolérance moyenne (mutations acceptées par acide aminé) pour chaque protéine.
  - `protein_tolerance_errorbars.png` : moyenne ± IC95% sur plusieurs runs afin de visualiser la variabilité inter-simulation.
  - `protein_tolerance_scatter.png` : graphique longueur vs tolérance, coloré par BLOSUM moyen et pondéré par la variation d’hydrophobicité.
  - `protein_accept_counts.tsv` : cumul des mutations acceptées, moyenne par run et tolérance par protéine.
  - `protein_tolerance_stats.tsv` : statistiques détaillées (σ, IC95, BLOSUM moyen, Δhydro moyen, probabilité d’acceptation).
  - `category_tolerance_summary.tsv` : synthèse par fonction (réplication, structurale, accessoire, non catégorisée).
- Analyse MVP `Mutation/analyze_mutations.py` : à partir du CSV des mutations acceptées, fournit un tableau de synthèse, des graphiques (histogramme de gravité, heatmap BLOSUM) et un score de risque moyen par protéine.
- Les exports critiques de l’analyse peuvent générer un CSV et un FASTA de mutations `critique` avec leur contexte protéique (fenêtre ±N aa) pour valider biologiquement les zones sensibles.
- Les annotations optionnelles (`--annotation annotations.csv`) permettent d’associer des protéines nommées ou de regrouper les positions non annotées.
- Les nouveaux paramètres de `simulate_mutations.py` couvrent le biais transition/transversion, les hotspots CpG, le poids d’usage des codons, la force de sélection ou la probabilité d’indels in-frame.
- Les graphiques et tableaux sont agnostiques au jeu de données : tout FASTA compatible avec Biopython est accepté. Les regroupements fonctionnels se basent sur des mots-clés (ex. `nsp`, `spike`, `membrane`), libre à vous d’ajuster le mapping dans `categorize_protein`.

## Installation & utilisation
- Prérequis : Python 3.13 et [uv](https://github.com/astral-sh/uv) disponibles dans le PATH.
- Cloner le dépôt puis installer les dépendances :  
  ```bash
  uv sync
  ```
- Lancer une simulation de mutations (exemple reproductible) :  
  ```bash
  uv run python Mutation/simulate_mutations.py --seed 42 --mutations 3 --samples 5 \
      --fasta Start/sequences.fasta --annotation annotations.csv
  ```
  - `--fasta` : remplace la séquence par un autre FASTA (mono ou multi-entrée).
  - `--record-id` : cible une séquence particulière si le FASTA en contient plusieurs.
  - `--annotation` : fichier `start,end,name[,product]` pour nommer les protéines (sinon, les régions non annotées sont regroupées).
  - `--mutations-csv` : enregistre un CSV `sample,protein,position,aa_ref,aa_mut,blosum,grantham,p_accept,severity,…` pour brancher directement l’analyse.
- Générer les graphiques et tableaux à partir du log produit (`out/metrics/batch.log` par défaut) :  
  ```bash
  python Mutation/generate_heatmap.py \
      --log out/metrics/batch.log \
      --fasta Start/sequences.fasta \
      --annotation annotations.csv \
      --output-dir out/metrics
  ```
  - Adapter `--log` si vous conservez plusieurs runs (un log par configuration est recommandé).
  - Les images/TSV sont régénérés à chaque exécution ; pensez à versionner les artefacts utiles.
- Analyser rapidement les mutations acceptées avec le MVP :  
  ```bash
  uv run python Mutation/analyze_mutations.py results/mutations_summary.csv \
      --summary-csv results/impact_summary.csv \
      --risk-csv results/oncogene_scores.csv \
      --barplot results/impact_bar.png \
      --heatmap results/blosum_heatmap.png \
      --fasta Start/sequences.fasta \
      --critical-csv results/mutations_critiques.csv \
      --critical-fasta results/mutations_critiques.fasta \
      --context-window 12
  ```
  - `--summary-csv` et `--risk-csv` exportent les tableaux agrégés.
  - `--barplot` / `--heatmap` sauvegardent les figures (PNG conseillé).
  - `--fasta` (et éventuellement `--record-id`, `--annotation`) reconstituent la séquence de référence pour extraire le contexte.
  - `--critical-csv` / `--critical-fasta` produisent un listing trié des mutations critiques avec flags, scores, positions et séquence locale.
  - `--context-window` ajuste la largeur du contexte (par défaut ±10 aa) ; `--show` ouvre les figures à l’écran.

### Workflow complet (exemple TP53)
1. **Simulation et enregistrement du log/CSV**  
   ```bash
   uv run python Mutation/simulate_mutations.py \
       --fasta Start/TP53.1.fasta \
       --mutations 100 \
       --samples 5 \
       --seed 42 \
       --preview-aa 0 \
       --mutations-csv results/tp53_mutations_summary.csv \
       | tee out/metrics/batch.log
   ```
   - Sorties: log détaillé `out/metrics/batch.log`, mutations acceptées `results/tp53_mutations_summary.csv`.

2. **Génération des heatmaps et tableaux agrégés**  
   ```bash
   uv run python Mutation/generate_heatmap.py \
       --log out/metrics/batch.log \
       --fasta Start/TP53.1.fasta \
       --output-dir out/metrics
   ```
   - Sorties (écrasées à chaque run): `out/metrics/protein_accept_heatmap.png`, `protein_tolerance_errorbars.png`, `protein_tolerance_scatter.png`, `protein_accept_counts.tsv`, `protein_tolerance_stats.tsv`, `category_tolerance_summary.tsv`.

3. **Analyse approfondie et exports critiques**  
   ```bash
   uv run python Mutation/analyze_mutations.py results/tp53_mutations_summary.csv \
       --fasta Start/TP53.1.fasta \
       --summary-csv results/tp53_impact_summary.csv \
       --risk-csv results/tp53_oncogene_scores.csv \
       --barplot results/tp53_impact_bar.png \
       --heatmap results/tp53_blosum_heatmap.png \
       --critical-csv results/tp53_mutations_critiques.csv \
       --critical-fasta results/tp53_mutations_critiques.fasta \
       --context-window 12
   ```
   - Sorties: `results/tp53_impact_summary.csv`, `results/tp53_oncogene_scores.csv`, `results/tp53_impact_bar.png`, `results/tp53_blosum_heatmap.png`, `results/tp53_mutations_critiques.csv`, `results/tp53_mutations_critiques.fasta`.

## Contribution
- Respecter PEP 8 (`flake8` peut être utilisé localement) et documenter les hypothèses biologiques dans les docstrings.
- Ajouter vos tests dans `tests/` (fixtures FASTA spécifiques bienvenues pour les scénarios unitaires).
- Avant une PR :
  1. `uv run python -m pytest`
  2. Inclure les commandes exécutées et l’impact observé (diff de log, aperçu de figure).
- Les jeux de données volumineux ou sensibles ne doivent pas être commités ; mentionner l’origine ou la procédure d’accès.
- Commits à l’impératif (ex. `Document mutation options`) avec lignes de description ≤ 72 caractères.
