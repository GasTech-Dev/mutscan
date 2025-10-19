"""Analyse and visualise mutation severities exported by the simulator."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __package__ in (None, ""):
    # Allow execution via ``python Mutation/analyze_mutations.py`` by exposing the package root.
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from Mutation.simulate_mutations import (
    build_feature_lookup,
    describe_feature,
    load_features,
    load_record,
    prepare_coding_sequence,
    sanitize_identifier,
)

REQUIRED_COLUMNS: set[str] = {
    "protein",
    "position",
    "aa_ref",
    "aa_mut",
    "blosum",
    "grantham",
    "p_accept",
    "severity",
}

IMPACT_ORDER: tuple[str, ...] = ("critique", "drastique", "conservatrice", "synonyme")
IMPACT_PALETTE: dict[str, str] = {
    "conservatrice": "#6ab04c",
    "drastique": "#f6e58d",
    "critique": "#eb4d4b",
    "synonyme": "#95afc0",
}
IMPACT_SCORES: dict[str, int] = {"critique": 3, "drastique": 2, "conservatrice": 1, "synonyme": 0}


def extract_context(sequence: str, position: int, window: int) -> str:
    """Return a short amino-acid window around the mutation (1-based position)."""

    if position <= 0:
        return ""
    index = position - 1
    start = max(0, index - window)
    end = min(len(sequence), index + window + 1)
    return sequence[start:end]


def build_reference_proteins(
    fasta_path: Path,
    record_id: str | None,
    annotation_path: Path | None,
) -> tuple[dict[str, str], dict[str, str], str]:
    """Return protein sequences and labels keyed by feature identifiers."""

    sequence_record = load_record(fasta_path, record_id)
    coding_seq = prepare_coding_sequence(sequence_record.seq)
    protein_seq = str(coding_seq.translate())

    features = load_features(annotation_path, sequence_record)
    sequences: dict[str, str] = {"non_annotée": protein_seq}
    display_names: dict[str, str] = {"non_annotée": describe_feature(None)}

    if features:
        lookup = build_feature_lookup(features, len(coding_seq))
        residues_by_feature: defaultdict[str, list[str]] = defaultdict(list)
        for feature in features:
            display_names[feature.name] = describe_feature(feature)
        for codon_index, residue in enumerate(protein_seq):
            start = codon_index * 3
            slice_features = lookup[start : start + 3]
            counts = Counter(candidate for candidate in slice_features if candidate is not None)
            feature = counts.most_common(1)[0][0] if counts else None
            feature_key = feature.name if feature else "non_annotée"
            residues_by_feature[feature_key].append(residue)
        for feature_key, residues in residues_by_feature.items():
            sequences[feature_key] = "".join(residues)

    prefix = sanitize_identifier(sequence_record.id or "mutation")
    return sequences, display_names, prefix


def classify_impact(severity: float, blosum: float) -> str:
    """Return the qualitative impact category for a mutation."""

    severity_value = int(severity)
    if severity_value >= 2 or blosum < -3.0:
        return "critique"
    if severity_value == 1 or blosum < -1.0:
        return "drastique"
    if severity_value == 0:
        return "conservatrice"
    return "synonyme"


def aggregate_impacts(mutations: pd.DataFrame) -> pd.DataFrame:
    """Summarise mutation impacts with basic descriptive statistics."""

    aggregated = (
        mutations.groupby("impact", observed=True)
        .agg(
            count=("impact", "size"),
            mean_blosum=("blosum", "mean"),
            mean_grantham=("grantham", "mean"),
            mean_accept=("p_accept", "mean"),
        )
        .reset_index()
    )
    aggregated["impact"] = pd.Categorical(aggregated["impact"], categories=IMPACT_ORDER, ordered=True)
    return aggregated.sort_values("impact").reset_index(drop=True)


def compute_oncogene_risk(mutations: pd.DataFrame) -> pd.Series:
    """Return a per-protein oncogenic hazard score based on impact weighting."""

    mapped_scores = mutations["impact"].map(IMPACT_SCORES)
    scores = (
        pd.to_numeric(mapped_scores, errors="coerce")
        .fillna(0.0)
        .astype(float)
    )
    return (
        mutations.assign(oncogene_score=scores)
        .groupby("protein", observed=True)["oncogene_score"]
        .mean()
        .sort_values(ascending=False)
    )


def plot_impact_summary(summary: pd.DataFrame) -> plt.Figure:
    """Return a barplot figure showing mutation counts per impact category."""

    present_categories = [category for category in IMPACT_ORDER if category in summary["impact"].astype(str).values]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=summary,
        x="impact",
        y="count",
        order=present_categories,
        palette=IMPACT_PALETTE,
        ax=ax,
    )
    ax.set_title("Répartition des mutations par gravité")
    ax.set_xlabel("Type de mutation")
    ax.set_ylabel("Nombre de mutations")
    ax.set_ylim(0, max(summary["count"].max(), 1))
    return fig


def plot_blosum_heatmap(mutations: pd.DataFrame) -> plt.Figure | None:
    """Return a heatmap figure showing mean BLOSUM by impact and protein."""

    pivot = (
        mutations.pivot_table(
            values="blosum",
            index="protein",
            columns="impact",
            aggfunc="mean",
        )
        .reindex(columns=IMPACT_ORDER)
        .dropna(how="all")
    )
    if pivot.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, max(3.5, 0.6 * len(pivot.index))))
    sns.heatmap(pivot, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("BLOSUM moyen par type de mutation et protéine")
    ax.set_xlabel("Type de mutation")
    ax.set_ylabel("Protéine")
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse les mutations simulées et génère des visualisations.")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("mutations_summary.csv"),
        help="Chemin du CSV des mutations exportées (défaut: mutations_summary.csv).",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Enregistre le tableau agrégé des impacts dans un CSV.",
    )
    parser.add_argument(
        "--risk-csv",
        type=Path,
        help="Enregistre l'indice de risque moyen par protéine dans un CSV.",
    )
    parser.add_argument(
        "--barplot",
        type=Path,
        help="Chemin du fichier image pour la répartition des impacts (PNG recommandé).",
    )
    parser.add_argument(
        "--heatmap",
        type=Path,
        help="Chemin du fichier image pour la heatmap BLOSUM (PNG recommandé).",
    )
    parser.add_argument(
        "--critical-csv",
        type=Path,
        help="Enregistre les mutations critiques enrichies avec un contexte local dans un CSV.",
    )
    parser.add_argument(
        "--critical-fasta",
        type=Path,
        help="Enregistre les mutations critiques (contexte ±N aa) dans un FASTA.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=10,
        help="Taille de la fenêtre en acides aminés pour le contexte (défaut: 10).",
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        help="Fichier FASTA de référence utilisé pour la simulation (nécessaire pour le contexte).",
    )
    parser.add_argument(
        "--record-id",
        help="Identifiant exact ou préfixe d'une séquence multi-FASTA pour reconstituer le contexte.",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        help="Fichier d'annotation (start,end,name[,product]) pour mapper les séquences protéiques.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Affiche les figures après rendu (défaut: enregistre uniquement si des chemins sont fournis).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Résolution (DPI) pour les figures sauvegardées (défaut: 150).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Fichier introuvable: {args.input}")
    if args.context_window < 0:
        raise SystemExit("--context-window doit être un entier positif ou nul.")

    df = pd.read_csv(args.input)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise SystemExit(f"Colonnes manquantes dans le CSV: {missing_list}")

    try:
        df["severity"] = pd.to_numeric(df["severity"], errors="raise")
        df["blosum"] = pd.to_numeric(df["blosum"], errors="raise")
        df["grantham"] = pd.to_numeric(df["grantham"], errors="raise")
        df["p_accept"] = pd.to_numeric(df["p_accept"], errors="raise")
    except ValueError as exc:  # pragma: no cover - message utilisateur
        raise SystemExit(f"Colonnes numériques invalides: {exc}") from exc

    df["impact"] = df.apply(lambda row: classify_impact(row["severity"], row["blosum"]), axis=1)
    df["impact"] = pd.Categorical(df["impact"], categories=IMPACT_ORDER, ordered=True)

    if df.empty:
        print("Aucune mutation à analyser.")
        return

    sns.set(style="whitegrid")

    summary = aggregate_impacts(df)
    if summary.empty:
        print("Aucune mutation classée pour le résumé.")
    else:
        print(summary.to_string(index=False))

    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_csv, index=False)

    risk = compute_oncogene_risk(df)
    if not risk.empty:
        print("\nIndice de risque par protéine:")
        print(risk.to_string())
    else:
        print("\nAucune protéine évaluée (liste vide).")

    if args.risk_csv:
        args.risk_csv.parent.mkdir(parents=True, exist_ok=True)
        risk.to_csv(args.risk_csv, header=["oncogene_score"])

    critical_exports_requested = args.critical_csv or args.critical_fasta
    if critical_exports_requested:
        if args.fasta is None:
            raise SystemExit(
                "Les options --critical-csv/--critical-fasta nécessitent un FASTA de référence (--fasta)."
            )
        sequences, display_names, prefix = build_reference_proteins(
            args.fasta,
            args.record_id,
            args.annotation,
        )
        window = int(args.context_window)
        protein_key_column = "protein_id" if "protein_id" in df.columns else "protein"
        sort_columns = [column for column in ("blosum", "position", "sample") if column in df.columns]
        dedupe_subset = [column for column in ("protein_id", "protein") if column in df.columns]
        dedupe_subset.extend(column for column in ("position", "aa_ref", "aa_mut") if column in df.columns)
        critical = (
            df[df["impact"] == "critique"]
            .sort_values(sort_columns)
            .drop_duplicates(subset=dedupe_subset, keep="first")
            .reset_index(drop=True)
        )
        if critical.empty:
            print("\nAucune mutation critique à exporter.")
        else:
            contexts: list[str] = []
            sequence_labels: list[str] = []
            for _, row in critical.iterrows():
                try:
                    position_value = int(row["position"])
                except (TypeError, ValueError):
                    position_value = 0
                protein_key = row.get(protein_key_column, "non_annotée") or "non_annotée"
                sequence = sequences.get(protein_key)
                if sequence is None:
                    # Recherche via le libellé descriptif si nécessaire.
                    label = str(row.get("protein") or "")
                    matched_keys = [key for key, name in display_names.items() if name == label]
                    for candidate in matched_keys:
                        sequence = sequences.get(candidate)
                        if sequence:
                            break
                if sequence is None:
                    sequence = sequences.get("non_annotée", "")
                contexts.append(extract_context(sequence, position_value, window))
                sequence_labels.append(protein_key)

            critical["sequence_context"] = contexts
            critical["protein_source"] = sequence_labels
            critical["mutation_id"] = [
                f"{prefix}_{index + 1:03d}" for index in range(len(critical))
            ]
            if "flags" in critical.columns:
                critical["flags"] = critical["flags"].fillna("")

            export_columns = [
                "mutation_id",
                "sample",
                "protein",
                "protein_id",
                "protein_source",
                "position",
                "position_nt",
                "aa_ref",
                "aa_mut",
                "impact",
                "severity",
                "blosum",
                "grantham",
                "p_accept",
                "flags",
                "sequence_context",
            ]
            available_columns = [column for column in export_columns if column in critical.columns]

            if args.critical_csv:
                args.critical_csv.parent.mkdir(parents=True, exist_ok=True)
                critical[available_columns].to_csv(args.critical_csv, index=False)
                print(f"\nMutations critiques exportées dans {args.critical_csv}")

            if args.critical_fasta:
                args.critical_fasta.parent.mkdir(parents=True, exist_ok=True)
                with args.critical_fasta.open("w") as handle:
                    for _, row in critical.iterrows():
                        context = row.get("sequence_context", "")
                        if not context:
                            continue
                        blosum_value = float(row.get("blosum", 0.0))
                        impact_label = str(row.get("impact", "NA"))
                        aa_ref = str(row.get("aa_ref", ""))
                        aa_mut = str(row.get("aa_mut", ""))
                        try:
                            position_label = int(row.get("position", 0))
                        except (TypeError, ValueError):
                            position_label = 0
                        header = (
                            f">{row['mutation_id']}|pos{position_label}"
                            f"|{aa_ref}>{aa_mut}|BLOSUM{blosum_value:+.1f}|impact={impact_label}"
                        )
                        handle.write(f"{header}\n{context}\n")
                print(f"Mutations critiques enregistrées dans {args.critical_fasta}")

    figures: list[tuple[str, plt.Figure]] = []
    if not summary.empty:
        figures.append(("barplot", plot_impact_summary(summary)))
    heatmap_fig = plot_blosum_heatmap(df)
    if heatmap_fig is not None:
        figures.append(("heatmap", heatmap_fig))

    for label, figure in figures:
        destination = args.barplot if label == "barplot" else args.heatmap
        if destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(destination, dpi=args.dpi, bbox_inches="tight")

    if args.show and figures:
        plt.show()
    else:
        for _, figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    main()
