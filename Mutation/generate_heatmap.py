"""Generate aggregated mutation plots and tables from simulation logs."""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __package__ in (None, ""):
    # Allow execution via ``python Mutation/generate_heatmap.py`` by exposing the package root.
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from Mutation.simulate_mutations import (
    describe_feature,
    load_features,
    load_record,
    prepare_coding_sequence,
)

SIMULATION_HEADER_PATTERN = re.compile(r"^Simulation\s+(\d+)")
LOG_LINE_PATTERN = re.compile(
    r"\[(?P<status>acceptée|rejetée)\].*?nt (?P<nt_position>\d+).*?codon (?P<codon_index>\d+):"
    r".*?aa (?P<aa_ref>[A-Z\*])(?:->|=)(?P<aa_mut>[A-Z\*]).*?score (?P<severity>-?\d+)\s*\("
    r"(?P<severity_label>[^)]+)\).*?BLOSUM (?P<blosum>[\-\d\.]+).*?Grantham (?P<grantham>[\d\.]+)"
    r".*?Δhydro (?P<hydro>[+\-\d\.]+).*?p_accept (?P<p_accept>[\d\.]+).*?protéine[:\s]*(?P<protein>.+?)\s*$"
)
DEFAULT_HEATMAP_NAME = "protein_accept_heatmap.png"
DEFAULT_COUNTS_NAME = "protein_accept_counts.tsv"
DEFAULT_ERRORBARS_NAME = "protein_tolerance_errorbars.png"
DEFAULT_SCATTER_NAME = "protein_tolerance_scatter.png"
DEFAULT_TOLERANCE_STATS_NAME = "protein_tolerance_stats.tsv"
DEFAULT_CATEGORY_SUMMARY_NAME = "category_tolerance_summary.tsv"
DEFAULT_DENSITY_NAME = "gene_mutation_density.png"
DEFAULT_DOMAIN_SUMMARY_NAME = "domain_tolerance_summary.tsv"
DEFAULT_INTERGENE_SUMMARY_NAME = "intergene_tolerance_summary.tsv"
DEFAULT_INTERGENE_FIGURE_NAME = "intergene_tolerance_comparison.png"


@dataclass(slots=True)
class MutationRecord:
    status: str
    protein: str
    blosum: float
    p_accept: float
    grantham: float
    hydrophobicity_delta: float
    simulation: int
    nt_position: int | None
    codon_index: int | None
    severity: int | None
    severity_label: str | None
    aa_reference: str | None
    aa_mutated: str | None

    @property
    def is_accepted(self) -> bool:
        return self.status == "acceptée"

    @property
    def amino_position(self) -> int | None:
        if self.codon_index is None:
            return None
        return self.codon_index + 1


@dataclass(slots=True, frozen=True)
class DomainDefinition:
    name: str
    start: int
    end: int
    protein: str | None = None
    product: str | None = None

    @property
    def length(self) -> int:
        return max(0, self.end - self.start + 1)

    def contains(self, amino_position: int) -> bool:
        return self.start <= amino_position <= self.end


def parse_log_lines(lines: Iterable[str]) -> list[MutationRecord]:
    """Extract mutation records from simulator output."""

    records: list[MutationRecord] = []
    current_simulation = 1
    for raw_line in lines:
        stripped = raw_line.strip()
        header_match = SIMULATION_HEADER_PATTERN.match(stripped)
        if header_match:
            try:
                current_simulation = int(header_match.group(1))
            except ValueError:
                current_simulation = 1

        match = LOG_LINE_PATTERN.search(raw_line)
        if not match:
            continue
        groups = match.groupdict()
        status = groups.get("status", "").strip()
        protein = (groups.get("protein") or "").strip()
        blosum = groups.get("blosum")
        grantham = groups.get("grantham")
        hydro_delta = groups.get("hydro")
        p_accept = groups.get("p_accept")
        nt_position_raw = groups.get("nt_position")
        codon_index_raw = groups.get("codon_index")
        severity_raw = groups.get("severity")
        severity_label = (groups.get("severity_label") or "").strip() or None
        aa_ref = (groups.get("aa_ref") or "").strip() or None
        aa_mut = (groups.get("aa_mut") or "").strip() or None

        try:
            blosum_value = float(blosum) if blosum is not None else float("nan")
            grantham_value = float(grantham) if grantham is not None else float("nan")
            hydro_value = float(hydro_delta) if hydro_delta is not None else float("nan")
            p_accept_value = float(p_accept) if p_accept is not None else float("nan")
        except ValueError:
            continue

        try:
            nt_position = int(nt_position_raw)
        except (TypeError, ValueError):
            nt_position = None

        try:
            codon_index = int(codon_index_raw)
        except (TypeError, ValueError):
            codon_index = None

        try:
            severity = int(severity_raw)
        except (TypeError, ValueError):
            severity = None

        record = MutationRecord(
            status=status,
            protein=protein,
            blosum=blosum_value,
            p_accept=p_accept_value,
            grantham=grantham_value,
            hydrophobicity_delta=hydro_value,
            simulation=current_simulation,
            nt_position=nt_position,
            codon_index=codon_index,
            severity=severity,
            severity_label=severity_label,
            aa_reference=aa_ref,
            aa_mutated=aa_mut,
        )
        records.append(record)
    return records


def load_domain_definitions(path: Path | None) -> dict[str | None, list[DomainDefinition]]:
    """Return domain definitions grouped by protein label."""

    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Fichier domaine introuvable: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Le fichier d'annotation des domaines est vide.")
        required = {"start", "end", "name"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Colonnes manquantes dans {path}: {', '.join(sorted(missing))}. "
                "Attendu: start,end,name[,protein,product]"
            )
        definitions: dict[str | None, list[DomainDefinition]] = defaultdict(list)
        for row in reader:
            try:
                start = int(row["start"])
                end = int(row["end"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Positions invalides pour le domaine '{row.get('name')}' dans {path}: "
                    f"start={row.get('start')}, end={row.get('end')}"
                ) from exc
            if start <= 0 or end <= 0:
                raise ValueError(
                    f"Les positions doivent être positives (domaine '{row.get('name')}' dans {path})."
                )
            if end < start:
                start, end = end, start
            name = (row.get("name") or "").strip()
            if not name:
                raise ValueError(f"Nom manquant pour un domaine dans {path}.")
            protein = (row.get("protein") or "").strip() or None
            product = (row.get("product") or "").strip() or None
            definitions[protein].append(DomainDefinition(name=name, start=start, end=end, protein=protein, product=product))

    for key, domain_list in definitions.items():
        domain_list.sort(key=lambda domain: (domain.start, domain.end))
    return dict(definitions)


def resolve_domain(
    domains_by_protein: Mapping[str | None, Sequence[DomainDefinition]],
    protein_label: str,
    amino_position: int,
) -> DomainDefinition | None:
    """Return the matching domain definition for a mutation."""

    candidates = domains_by_protein.get(protein_label, ())
    for domain in candidates:
        if domain.contains(amino_position):
            return domain
    global_candidates = domains_by_protein.get(None, ())
    for domain in global_candidates:
        if domain.contains(amino_position):
            return domain
    return None


def count_acceptances(records: Sequence[MutationRecord]) -> Counter[str]:
    """Return the number of accepted mutations per protein."""

    counter: Counter[str] = Counter()
    for record in records:
        if record.is_accepted:
            counter[record.protein] += 1
    return counter


def records_to_dataframe(records: Sequence[MutationRecord]) -> pd.DataFrame:
    """Convert raw mutation records into a pandas DataFrame."""

    if not records:
        return pd.DataFrame(
            columns=[
                "status",
                "protein",
                "blosum",
                "p_accept",
                "grantham",
                "hydrophobicity_delta",
                "simulation",
                "nt_position",
                "codon_index",
                "severity",
                "severity_label",
                "aa_reference",
                "aa_mutated",
            ]
        )

    return pd.DataFrame(
        {
            "status": [record.status for record in records],
            "protein": [record.protein for record in records],
            "blosum": [record.blosum for record in records],
            "p_accept": [record.p_accept for record in records],
            "grantham": [record.grantham for record in records],
            "hydrophobicity_delta": [record.hydrophobicity_delta for record in records],
            "simulation": [record.simulation for record in records],
            "nt_position": [record.nt_position for record in records],
            "codon_index": [record.codon_index for record in records],
            "severity": [record.severity for record in records],
            "severity_label": [record.severity_label for record in records],
            "aa_reference": [record.aa_reference for record in records],
            "aa_mutated": [record.aa_mutated for record in records],
        }
    )


def build_tolerance_per_run(
    records_frame: pd.DataFrame,
    lengths: Mapping[str, float],
) -> pd.DataFrame:
    """Return tolerance values (mut./aa) per protein and per simulation run."""

    if not lengths:
        return pd.DataFrame(
            columns=["simulation", "protein", "accepted", "tolerance", "protein_length_aa"]
        )

    proteins = sorted(lengths.keys())

    if records_frame.empty:
        runs = [1]
    else:
        runs = sorted({int(value) for value in records_frame["simulation"].unique()})
    if not runs:
        runs = [1]

    accepted_frame = records_frame[records_frame["status"] == "acceptée"]
    if accepted_frame.empty:
        accepted_counts: dict[tuple[int, str], int] = {}
    else:
        accepted_counts = (
            accepted_frame.groupby(["simulation", "protein"])
            .size()
            .astype(int)
            .to_dict()
        )

    rows: list[dict[str, float | int | str]] = []
    for run in runs:
        for protein in proteins:
            length = lengths.get(protein)
            if not length or length <= 0:
                continue
            accepted = accepted_counts.get((run, protein), 0)
            tolerance = accepted / length
            rows.append(
                {
                    "simulation": run,
                    "protein": protein,
                    "accepted": accepted,
                    "tolerance": tolerance,
                    "protein_length_aa": length,
                }
            )

    return pd.DataFrame(rows)


def compute_protein_statistics(
    per_run_frame: pd.DataFrame,
    records_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate tolerance, BLOSUM and hydrophobicity metrics per protein."""

    if per_run_frame.empty:
        return pd.DataFrame(
            columns=[
                "mean_tolerance",
                "std_tolerance",
                "ci95_tolerance",
                "runs",
                "protein_length_aa",
                "total_accepted",
                "mean_blosum",
                "mean_hydrophobicity_delta",
                "mean_acceptance_probability",
            ]
        )

    stats = (
        per_run_frame.groupby("protein")
        .agg(
            mean_tolerance=("tolerance", "mean"),
            std_tolerance=("tolerance", "std"),
            runs=("tolerance", "count"),
            protein_length_aa=("protein_length_aa", "first"),
        )
        .sort_index()
    )
    stats["std_tolerance"] = stats["std_tolerance"].fillna(0.0)
    stats["runs"] = stats["runs"].astype(int)
    ci95 = 1.96 * stats["std_tolerance"] / stats["runs"].clip(lower=1).pow(0.5)
    ci95 = ci95.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    stats["ci95_tolerance"] = ci95
    stats.loc[stats["runs"] <= 1, "ci95_tolerance"] = 0.0

    accepted_frame = records_frame[records_frame["status"] == "acceptée"]
    if accepted_frame.empty:
        accepted_counts = pd.Series(dtype=int)
        blosum_means = pd.Series(dtype=float)
        hydro_means = pd.Series(dtype=float)
        acceptance_means = pd.Series(dtype=float)
    else:
        accepted_counts = accepted_frame.groupby("protein").size()
        blosum_means = accepted_frame.groupby("protein")["blosum"].mean()
        hydro_means = accepted_frame.groupby("protein")["hydrophobicity_delta"].mean()
        acceptance_means = accepted_frame.groupby("protein")["p_accept"].mean()

    stats["total_accepted"] = accepted_counts.reindex(stats.index, fill_value=0).astype(int)
    stats["mean_blosum"] = blosum_means.reindex(stats.index)
    stats["mean_hydrophobicity_delta"] = hydro_means.reindex(stats.index)
    stats["mean_acceptance_probability"] = acceptance_means.reindex(stats.index)

    return stats


def write_tolerance_stats_table(stats: pd.DataFrame, destination: Path) -> None:
    """Persist detailed tolerance statistics (run mean, dispersion, CI)."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "protein",
        "runs",
        "mean_tolerance",
        "std_tolerance",
        "ci95_tolerance",
        "protein_length_aa",
        "total_accepted",
        "mean_blosum",
        "mean_hydrophobicity_delta",
        "mean_acceptance_probability",
    ]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        if stats.empty:
            return
        for protein, row in stats.reset_index().set_index("protein").iterrows():
            writer.writerow(
                {
                    "protein": protein,
                    "runs": int(row["runs"]),
                    "mean_tolerance": f"{row['mean_tolerance']:.6f}",
                    "std_tolerance": f"{row['std_tolerance']:.6f}",
                    "ci95_tolerance": f"{row['ci95_tolerance']:.6f}",
                    "protein_length_aa": f"{row['protein_length_aa']:.2f}",
                    "total_accepted": int(row["total_accepted"]),
                    "mean_blosum": "" if pd.isna(row["mean_blosum"]) else f"{row['mean_blosum']:.3f}",
                    "mean_hydrophobicity_delta": ""
                    if pd.isna(row["mean_hydrophobicity_delta"])
                    else f"{row['mean_hydrophobicity_delta']:.3f}",
                    "mean_acceptance_probability": ""
                    if pd.isna(row["mean_acceptance_probability"])
                    else f"{row['mean_acceptance_probability']:.3f}",
                }
            )


def render_tolerance_errorbars(stats: pd.DataFrame, output_path: Path, *, dpi: int) -> None:
    """Plot mean tolerance with 95% confidence intervals across runs."""

    if stats.empty:
        print("Aucune donnée pour générer les barres d'erreur de tolérance.")
        return
    if stats["runs"].max() <= 1:
        print("Une seule simulation détectée; barres d'erreur non générées.")
        return

    ordered = stats.sort_values("mean_tolerance", ascending=False)
    indices = range(len(ordered))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, len(ordered) * 0.4), 6))
    ax.errorbar(
        list(indices),
        ordered["mean_tolerance"],
        yerr=ordered["ci95_tolerance"],
        fmt="o",
        ecolor="tab:blue",
        color="tab:blue",
        capsize=4,
        elinewidth=1,
        markersize=5,
    )
    ax.set_xticks(list(indices))
    ax.set_xticklabels(ordered.index, rotation=90)
    ax.set_ylabel("Tolérance moyenne (mut./aa)")
    ax.set_xlabel("Protéine")
    ax.set_title("Tolérance moyenne ± IC95% par protéine")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Barres d'erreur enregistrées dans {output_path}")


def render_correlation_scatter(
    stats: pd.DataFrame,
    output_path: Path,
    *,
    dpi: int,
    cmap: str,
) -> None:
    """Render tolerance vs length scatter plot coloured by mean BLOSUM."""

    if stats.empty:
        print("Aucune donnée pour générer le nuage de points tolérance/longueur.")
        return

    subset = stats.dropna(subset=["mean_blosum"])
    if subset.empty:
        print(
            "Impossible de générer le nuage de points : aucun BLOSUM moyen disponible (aucune mutation acceptée)."
        )
        return

    sizes = 40 + 160 * subset["mean_hydrophobicity_delta"].abs().fillna(0.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        subset["protein_length_aa"],
        subset["mean_tolerance"],
        c=subset["mean_blosum"],
        cmap=cmap,
        s=sizes,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlabel("Longueur protéique (aa)")
    ax.set_ylabel("Tolérance moyenne (mut./aa)")
    ax.set_title("Tolérance vs longueur (couleur = BLOSUM moyen, taille = |Δhydro| moyen)")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("BLOSUM moyen")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Nuage de points enregistré dans {output_path}")

    corr_length = subset["protein_length_aa"].corr(subset["mean_tolerance"])
    if not math.isnan(corr_length):
        print(f"Corrélation longueur/tolérance (Pearson): {corr_length:.3f}")
    corr_blosum = subset["mean_blosum"].corr(subset["mean_tolerance"])
    if not math.isnan(corr_blosum):
        print(f"Corrélation BLOSUM/tolérance (Pearson): {corr_blosum:.3f}")

    hydro_abs = subset["mean_hydrophobicity_delta"].abs()
    if not hydro_abs.isna().all():
        corr_hydro = hydro_abs.corr(subset["mean_tolerance"])
        if corr_hydro is not None and not math.isnan(corr_hydro):
            print(f"Corrélation |Δhydro|/tolérance (Pearson): {corr_hydro:.3f}")


def categorize_protein(label: str) -> str:
    """Heuristically map a protein label to a functional category."""

    lower = label.lower()
    if "non annot" in lower:
        return "non catégorisée"

    replication_keywords = (
        "nsp",
        "polymerase",
        "helicase",
        "methyltransferase",
        "protease",
        "endonuclease",
        "exoribonuclease",
        "replicase",
    )
    if any(keyword in lower for keyword in replication_keywords):
        return "réplication"

    structural_keywords = (
        "spike",
        "envelope",
        "membrane",
        "nucleocapsid",
        "capsid",
        "glycoprotein",
    )
    if any(keyword in lower for keyword in structural_keywords):
        return "structurale"

    return "accessoire"


def compute_category_summary(per_run_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tolerance statistics per functional category."""

    if per_run_frame.empty:
        return pd.DataFrame(
            columns=[
                "mean_tolerance",
                "std_tolerance",
                "ci95_tolerance",
                "runs",
                "accepted_total",
                "proteins",
                "mean_length_aa",
            ]
        )

    frame = per_run_frame.copy()
    frame["category"] = frame["protein"].map(categorize_protein)

    category_runs = (
        frame.groupby(["simulation", "category"])
        .agg(
            accepted=("accepted", "sum"),
            total_length=("protein_length_aa", "sum"),
        )
        .reset_index()
    )
    category_runs["tolerance"] = category_runs.apply(
        lambda row: row["accepted"] / row["total_length"] if row["total_length"] else 0.0,
        axis=1,
    )

    category_stats = (
        category_runs.groupby("category")
        .agg(
            mean_tolerance=("tolerance", "mean"),
            std_tolerance=("tolerance", "std"),
            runs=("tolerance", "count"),
            accepted_total=("accepted", "sum"),
            total_length_mean=("total_length", "mean"),
        )
        .sort_index()
    )
    category_stats["std_tolerance"] = category_stats["std_tolerance"].fillna(0.0)
    category_stats["runs"] = category_stats["runs"].astype(int)
    ci95 = 1.96 * category_stats["std_tolerance"] / category_stats["runs"].clip(lower=1).pow(0.5)
    ci95 = ci95.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    category_stats["ci95_tolerance"] = ci95
    category_stats.loc[category_stats["runs"] <= 1, "ci95_tolerance"] = 0.0

    proteins_per_category = frame.groupby("category")["protein"].nunique()
    category_stats["proteins"] = proteins_per_category.reindex(category_stats.index, fill_value=0)
    category_stats["mean_length_aa"] = (
        frame.groupby("category")["protein_length_aa"].mean().reindex(category_stats.index)
    )

    return category_stats


def write_category_summary(stats: pd.DataFrame, destination: Path) -> None:
    """Persist the category-level tolerance summary."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "runs",
        "proteins",
        "mean_tolerance",
        "std_tolerance",
        "ci95_tolerance",
        "accepted_total",
        "mean_length_aa",
    ]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        if stats.empty:
            return
        for category, row in stats.reset_index().set_index("category").iterrows():
            writer.writerow(
                {
                    "category": category,
                    "runs": int(row["runs"]),
                    "proteins": int(row["proteins"]),
                    "mean_tolerance": f"{row['mean_tolerance']:.6f}",
                    "std_tolerance": f"{row['std_tolerance']:.6f}",
                    "ci95_tolerance": f"{row['ci95_tolerance']:.6f}",
                    "accepted_total": int(row["accepted_total"]),
                    "mean_length_aa": ""
                    if pd.isna(row["mean_length_aa"])
                    else f"{row['mean_length_aa']:.2f}",
                }
            )


def compute_domain_summary(
    records: Sequence[MutationRecord],
    domains_by_protein: Mapping[str | None, Sequence[DomainDefinition]],
) -> pd.DataFrame:
    """Aggregate mutation metrics per functional domain."""

    columns = [
        "protein",
        "domain",
        "product",
        "start",
        "end",
        "length",
        "accepted",
        "mutations_per_aa",
        "mean_severity",
        "mean_blosum",
        "mean_grantham",
        "mean_acceptance_probability",
        "mean_hydrophobicity_delta",
    ]
    if not domains_by_protein:
        return pd.DataFrame(columns=columns)

    domain_lists = list(domains_by_protein.values())
    if not domain_lists:
        return pd.DataFrame(columns=columns)

    accepted_records = [
        record
        for record in records
        if record.is_accepted and record.amino_position is not None
    ]
    domain_buckets: dict[DomainDefinition, list[MutationRecord]] = defaultdict(list)
    for record in accepted_records:
        domain = resolve_domain(domains_by_protein, record.protein, record.amino_position)
        if domain is not None:
            domain_buckets[domain].append(record)

    all_domains: list[DomainDefinition] = []
    for definitions in domains_by_protein.values():
        all_domains.extend(definitions)

    rows: list[dict[str, object]] = []
    for domain in sorted(all_domains, key=lambda item: (item.protein or "", item.start, item.end)):
        bucket = domain_buckets.get(domain, [])
        accepted_count = len(bucket)
        length = domain.length or 0
        density = accepted_count / length if length > 0 else 0.0

        def mean_or_nan(values: Iterable[float | int | None]) -> float:
            filtered: list[float] = []
            for value in values:
                if value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isnan(numeric):
                    continue
                filtered.append(numeric)
            if not filtered:
                return float("nan")
            return float(sum(filtered) / len(filtered))

        rows.append(
            {
                "protein": domain.protein or "global",
                "domain": domain.name,
                "product": domain.product or "",
                "start": domain.start,
                "end": domain.end,
                "length": length,
                "accepted": accepted_count,
                "mutations_per_aa": density,
                "mean_severity": mean_or_nan(record.severity for record in bucket),
                "mean_blosum": mean_or_nan(record.blosum for record in bucket),
                "mean_grantham": mean_or_nan(record.grantham for record in bucket),
                "mean_acceptance_probability": mean_or_nan(record.p_accept for record in bucket),
                "mean_hydrophobicity_delta": mean_or_nan(record.hydrophobicity_delta for record in bucket),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def write_domain_summary(stats: pd.DataFrame, destination: Path) -> None:
    """Write domain-level tolerance summary to disk."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "protein",
        "domain",
        "product",
        "start",
        "end",
        "length",
        "accepted",
        "mutations_per_aa",
        "mean_severity",
        "mean_blosum",
        "mean_grantham",
        "mean_acceptance_probability",
        "mean_hydrophobicity_delta",
    ]

    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        if stats.empty:
            return
        for row in stats.itertuples(index=False):
            writer.writerow(
                {
                    "protein": row.protein,
                    "domain": row.domain,
                    "product": row.product,
                    "start": int(row.start),
                    "end": int(row.end),
                    "length": int(row.length),
                    "accepted": int(row.accepted),
                    "mutations_per_aa": f"{row.mutations_per_aa:.6f}",
                    "mean_severity": ""
                    if pd.isna(row.mean_severity)
                    else f"{row.mean_severity:.3f}",
                    "mean_blosum": ""
                    if pd.isna(row.mean_blosum)
                    else f"{row.mean_blosum:.3f}",
                    "mean_grantham": ""
                    if pd.isna(row.mean_grantham)
                    else f"{row.mean_grantham:.3f}",
                    "mean_acceptance_probability": ""
                    if pd.isna(row.mean_acceptance_probability)
                    else f"{row.mean_acceptance_probability:.3f}",
                    "mean_hydrophobicity_delta": ""
                    if pd.isna(row.mean_hydrophobicity_delta)
                    else f"{row.mean_hydrophobicity_delta:.3f}",
                }
            )


def compute_mutation_density_matrices(
    records: Sequence[MutationRecord],
    lengths: Mapping[str, float],
    *,
    bin_size: int,
) -> dict[str, pd.DataFrame]:
    """Return density matrices (accepted/rejected) per protein along the gene."""

    if bin_size <= 0:
        raise ValueError("La taille des fenêtres de densité doit être positive.")

    matrices: dict[str, pd.DataFrame] = {}
    for protein, length in lengths.items():
        length_int = int(round(length))
        if length_int <= 0:
            continue
        bin_count = max(1, math.ceil(length_int / bin_size))
        columns: list[str] = []
        for idx in range(bin_count):
            start = idx * bin_size + 1
            end = min(length_int, (idx + 1) * bin_size)
            columns.append(f"{start}-{end}")
        matrices[protein] = pd.DataFrame(
            0.0,
            index=["acceptées", "rejetées"],
            columns=columns,
        )

    if not matrices:
        return {}

    for record in records:
        amino_position = record.amino_position
        if amino_position is None:
            continue
        matrix = matrices.get(record.protein)
        if matrix is None or matrix.empty:
            continue
        bin_index = min((amino_position - 1) // bin_size, matrix.shape[1] - 1)
        row_label = "acceptées" if record.is_accepted else "rejetées"
        matrix.iat[matrix.index.get_loc(row_label), bin_index] += 1.0

    return matrices


def render_mutation_density_heatmaps(
    matrices: Mapping[str, pd.DataFrame],
    output_path: Path,
    *,
    dpi: int,
    cmap: str,
) -> None:
    """Render heatmaps highlighting mutation density along the gene."""

    if not matrices:
        print("Aucune donnée pour la heatmap de densité de gène.")
        return

    proteins = list(matrices.keys())
    figure_width = max(8.0, max(matrix.shape[1] * 0.6 for matrix in matrices.values()))
    figure_height = 3.5 * len(proteins)

    vmax_candidates = [
        float(matrix.values.max()) for matrix in matrices.values() if not matrix.values.size == 0
    ]
    global_vmax = max(vmax_candidates) if vmax_candidates else 0.0
    if math.isnan(global_vmax) or global_vmax <= 0:
        global_vmax = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(proteins), 1, figsize=(figure_width, figure_height), squeeze=False)
    for axis_index, (ax, protein) in enumerate(zip(axes.flat, proteins)):
        matrix = matrices[protein]
        if matrix.values.sum() == 0:
            sns.heatmap(matrix, ax=ax, cmap=cmap, cbar=False, annot=False, vmin=0.0, vmax=global_vmax)
            ax.text(
                0.5,
                0.5,
                "Aucune mutation",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )
        else:
            sns.heatmap(
                matrix,
                ax=ax,
                cmap=cmap,
                cbar=(axis_index == 0),
                linewidths=0.4,
                linecolor="white",
                vmin=0.0,
                vmax=global_vmax,
            )
        ax.set_title(f"Densité de mutations ({protein})")
        ax.set_xlabel("Fenêtre (aa)")
        ax.set_ylabel("Statut")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Heatmap de densité enregistrée dans {output_path}")


def compute_dataset_summary(
    per_run_frame: pd.DataFrame,
    records_frame: pd.DataFrame,
    lengths: Mapping[str, float],
) -> pd.Series:
    """Compute global tolerance metrics for a dataset."""

    total_length = float(sum(length for length in lengths.values() if length and length > 0))

    if per_run_frame.empty:
        runs = 0
        mean_tolerance = 0.0
        std_tolerance = 0.0
        ci95 = 0.0
        accepted_total = 0
    else:
        run_stats = (
            per_run_frame.groupby("simulation")
            .agg(
                accepted=("accepted", "sum"),
                total_length=("protein_length_aa", "sum"),
            )
            .reset_index(drop=True)
        )
        if run_stats.empty:
            runs = 0
            mean_tolerance = 0.0
            std_tolerance = 0.0
            ci95 = 0.0
            accepted_total = 0
        else:
            run_stats["total_length"] = run_stats["total_length"].replace(0, float("nan"))
            run_stats["tolerance"] = run_stats["accepted"] / run_stats["total_length"]
            tolerance_series = run_stats["tolerance"].dropna()
            runs = len(tolerance_series)
            if runs == 0:
                mean_tolerance = 0.0
                std_tolerance = 0.0
                ci95 = 0.0
            else:
                mean_tolerance = float(tolerance_series.mean())
                std_tolerance = float(tolerance_series.std(ddof=1)) if runs > 1 else 0.0
                ci95 = 1.96 * std_tolerance / math.sqrt(runs) if runs > 1 else 0.0
            accepted_total = int(run_stats["accepted"].sum())

    accepted_frame = records_frame[records_frame["status"] == "acceptée"]
    mean_blosum = float(accepted_frame["blosum"].mean()) if not accepted_frame.empty else float("nan")
    mean_acceptance = (
        float(accepted_frame["p_accept"].mean()) if not accepted_frame.empty else float("nan")
    )
    mean_grantham = (
        float(accepted_frame["grantham"].mean()) if not accepted_frame.empty else float("nan")
    )
    mean_hydro = (
        float(accepted_frame["hydrophobicity_delta"].mean())
        if not accepted_frame.empty
        else float("nan")
    )

    return pd.Series(
        {
            "runs": runs,
            "mean_tolerance": mean_tolerance,
            "std_tolerance": std_tolerance,
            "ci95_tolerance": ci95,
            "accepted_total": accepted_total,
            "total_length_aa": total_length,
            "mean_blosum": mean_blosum,
            "mean_acceptance_probability": mean_acceptance,
            "mean_grantham": mean_grantham,
            "mean_hydrophobicity_delta": mean_hydro,
        }
    )


def write_intergene_summary(summary: pd.DataFrame, destination: Path) -> None:
    """Write inter-gene comparison metrics to TSV."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "runs",
        "mean_tolerance",
        "std_tolerance",
        "ci95_tolerance",
        "accepted_total",
        "total_length_aa",
        "mean_blosum",
        "mean_acceptance_probability",
        "mean_grantham",
        "mean_hydrophobicity_delta",
    ]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        if summary.empty:
            return
        for row in summary.itertuples(index=False):
            writer.writerow(
                {
                    "label": row.label,
                    "runs": int(row.runs),
                    "mean_tolerance": f"{row.mean_tolerance:.6f}",
                    "std_tolerance": f"{row.std_tolerance:.6f}",
                    "ci95_tolerance": f"{row.ci95_tolerance:.6f}",
                    "accepted_total": int(row.accepted_total),
                    "total_length_aa": f"{row.total_length_aa:.2f}",
                    "mean_blosum": ""
                    if pd.isna(row.mean_blosum)
                    else f"{row.mean_blosum:.3f}",
                    "mean_acceptance_probability": ""
                    if pd.isna(row.mean_acceptance_probability)
                    else f"{row.mean_acceptance_probability:.3f}",
                    "mean_grantham": ""
                    if pd.isna(row.mean_grantham)
                    else f"{row.mean_grantham:.3f}",
                    "mean_hydrophobicity_delta": ""
                    if pd.isna(row.mean_hydrophobicity_delta)
                    else f"{row.mean_hydrophobicity_delta:.3f}",
                }
            )


def render_intergene_comparison_plot(summary: pd.DataFrame, output_path: Path, *, dpi: int) -> None:
    """Render a bar plot comparing tolerance between genes."""

    if summary.empty:
        print("Aucune donnée de comparaison inter-gènes à tracer.")
        return

    labels = summary["label"].tolist()
    values = summary["mean_tolerance"].tolist()
    errors = summary["ci95_tolerance"].fillna(0.0).tolist()
    palette = sns.color_palette("Set2", len(labels))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6.0, len(labels) * 1.6), 5.0))
    positions = range(len(labels))
    bars = ax.bar(positions, values, color=palette, edgecolor="black", linewidth=0.7)
    ax.errorbar(
        positions,
        values,
        yerr=errors,
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=4,
    )

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Tolérance moyenne (mut./aa)")
    ax.set_title("Comparaison inter-gènes de la tolérance mutationnelle")

    for bar, accepted in zip(bars, summary["accepted_total"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{accepted} mut.",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Comparaison inter-gènes enregistrée dans {output_path}")


def analyse_dataset(
    *,
    log_path: Path,
    fasta_path: Path,
    record_id: str | None,
    annotation_path: Path | None,
) -> tuple[
    list[MutationRecord],
    pd.DataFrame,
    Counter,
    dict[str, float],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
]:
    """Load, parse and aggregate simulation outputs for a dataset."""

    if not log_path.exists():
        raise FileNotFoundError(f"Le fichier de log {log_path} est introuvable.")
    records = parse_log_lines(log_path.read_text().splitlines())
    records_frame = records_to_dataframe(records)
    counts = count_acceptances(records)

    lengths = compute_protein_lengths(
        fasta_path=fasta_path,
        record_id=record_id,
        annotation_path=annotation_path,
    )
    per_run_frame = build_tolerance_per_run(records_frame, lengths)
    protein_stats = compute_protein_statistics(per_run_frame, records_frame)
    category_stats = compute_category_summary(per_run_frame)
    dataset_summary = compute_dataset_summary(per_run_frame, records_frame, lengths)
    return (
        records,
        records_frame,
        counts,
        lengths,
        per_run_frame,
        protein_stats,
        category_stats,
        dataset_summary,
    )


def parse_comparison_argument(argument: str) -> tuple[str, Path, Path, Path | None, str | None]:
    """Parse a comparison argument formatted as label|log|fasta[|annotation][|record_id]."""

    parts = [part.strip() for part in argument.split("|")]
    if len(parts) < 3:
        raise ValueError(
            "Chaque option --comparison doit suivre le format label|log|fasta[|annotation][|record_id]."
        )
    label = parts[0]
    if not label:
        raise ValueError("Le label de comparaison ne peut pas être vide.")
    log_path = Path(parts[1])
    fasta_path = Path(parts[2])
    annotation_path = Path(parts[3]) if len(parts) >= 4 and parts[3] else None
    record_id = parts[4] if len(parts) >= 5 and parts[4] else None
    return label, log_path, fasta_path, annotation_path, record_id


def write_counts_table(rows: Sequence[dict], destination: Path) -> None:
    """Persist the aggregated table in TSV format."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "protein",
                "accepted_mutations",
                "accepted_mutations_per_run",
                "protein_length_aa",
                "tolerance_rate",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_protein_lengths(
    *,
    fasta_path: Path,
    record_id: str | None,
    annotation_path: Path | None,
) -> dict[str, float]:
    """Return protein lengths (amino acids) using the same annotation logic as the simulator."""

    record = load_record(fasta_path, record_id)
    coding_seq = prepare_coding_sequence(record.seq)
    features = load_features(annotation_path, record)

    if not features:
        return {"région non annotée": len(coding_seq) / 3}

    lengths: dict[str, float] = {}
    coverage = [False] * len(coding_seq)

    for feature in features:
        start_index = max(feature.start - 1, 0)
        end_index = min(feature.end, len(coding_seq))
        if end_index <= start_index:
            continue
        length_nt = end_index - start_index
        length_aa = length_nt / 3
        label = describe_feature(feature)
        lengths[label] = lengths.get(label, 0.0) + length_aa
        for index in range(start_index, end_index):
            coverage[index] = True

    uncovered_nt = sum(1 for flag in coverage if not flag)
    if uncovered_nt:
        lengths["région non annotée"] = uncovered_nt / 3

    return lengths


def merge_counts_lengths(
    counts: Counter[str],
    lengths: dict[str, float],
    *,
    run_count: int = 1,
) -> list[dict]:
    """Combine counts and lengths, computing tolerance rates."""

    if run_count <= 0:
        run_count = 1

    rows: list[dict] = []
    for protein, total in sorted(counts.items(), key=lambda item: item[0]):
        length = lengths.get(protein)
        accepted_per_run = total / run_count
        if not length or length <= 0:
            tolerance_value: float | None = None
        else:
            tolerance_value = accepted_per_run / length
        rows.append(
            {
                "protein": protein,
                "accepted_mutations": total,
                "accepted_mutations_per_run": f"{accepted_per_run:.3f}",
                "protein_length_aa": f"{length:.2f}" if length else "",
                "tolerance_rate": (
                    f"{tolerance_value:.6f}" if tolerance_value is not None else ""
                ),
            }
        )
    return rows


def build_dataframe(
    counts: Counter[str],
    lengths: dict[str, float],
    *,
    run_count: int = 1,
) -> pd.DataFrame:
    """Create a dataframe with tolerance rates for heatmap rendering."""

    if run_count <= 0:
        run_count = 1

    records: list[dict[str, float]] = []
    for protein, length in lengths.items():
        if not length or length <= 0:
            continue
        total = counts.get(protein, 0)
        mean_accepted = total / run_count
        records.append(
            {
                "protein": protein,
                "accepted_mutations": total,
                "protein_length_aa": length,
                "tolerance_rate": mean_accepted / length,
            }
        )
    if not records:
        return pd.DataFrame(columns=["tolerance_rate"]).set_index("protein")
    frame = pd.DataFrame(records).set_index("protein").sort_index()
    return frame


def render_heatmap(frame: pd.DataFrame, output_path: Path, *, dpi: int, cmap: str) -> None:
    """Render and save the heatmap plot."""

    if frame.empty:
        print("Aucune mutation acceptée trouvée; heatmap non générée.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    values = frame[["tolerance_rate"]]
    plt.figure(figsize=(6, max(4, len(values) * 0.3)))
    sns.heatmap(values, annot=True, cmap=cmap, cbar=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Heatmap enregistrée dans {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Génère un heatmap de mutations acceptées par protéine à partir du log des simulations.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("out/metrics/batch.log"),
        help="Chemin du log produit par simulate_mutations.py (défaut: out/metrics/batch.log).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/metrics"),
        help="Répertoire où enregistrer les artefacts (défaut: out/metrics).",
    )
    parser.add_argument(
        "--heatmap-name",
        default=DEFAULT_HEATMAP_NAME,
        help=f"Nom du fichier image pour la heatmap (défaut: {DEFAULT_HEATMAP_NAME}).",
    )
    parser.add_argument(
        "--counts-name",
        default=DEFAULT_COUNTS_NAME,
        help=f"Nom du fichier TSV agrégé (défaut: {DEFAULT_COUNTS_NAME}).",
    )
    parser.add_argument(
        "--tolerance-stats-name",
        default=DEFAULT_TOLERANCE_STATS_NAME,
        help=f"Nom du TSV détaillant moyennes/IC (défaut: {DEFAULT_TOLERANCE_STATS_NAME}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Définition de l'image exportée (défaut: 300 DPI).",
    )
    parser.add_argument(
        "--cmap",
        default="mako",
        help="Palette de couleurs seaborn/matplotlib pour la heatmap (défaut: mako).",
    )
    parser.add_argument(
        "--errorbars-name",
        default=DEFAULT_ERRORBARS_NAME,
        help=f"Nom du fichier image pour les barres d'erreur (défaut: {DEFAULT_ERRORBARS_NAME}).",
    )
    parser.add_argument(
        "--scatter-name",
        default=DEFAULT_SCATTER_NAME,
        help=f"Nom du scatter plot longueur/tolérance (défaut: {DEFAULT_SCATTER_NAME}).",
    )
    parser.add_argument(
        "--category-summary-name",
        default=DEFAULT_CATEGORY_SUMMARY_NAME,
        help=f"Nom du TSV de synthèse fonctionnelle (défaut: {DEFAULT_CATEGORY_SUMMARY_NAME}).",
    )
    parser.add_argument(
        "--density-name",
        default=DEFAULT_DENSITY_NAME,
        help=f"Nom du fichier image pour la heatmap de densité (défaut: {DEFAULT_DENSITY_NAME}).",
    )
    parser.add_argument(
        "--density-bin",
        type=int,
        default=25,
        help="Taille des fenêtres (aa) pour la heatmap de densité (défaut: 25).",
    )
    parser.add_argument(
        "--density-cmap",
        default="rocket_r",
        help="Palette de couleurs pour la heatmap de densité (défaut: rocket_r).",
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=Path("Start/sequences.fasta"),
        help="FASTA de référence pour récupérer les longueurs protéiques (défaut: Start/sequences.fasta).",
    )
    parser.add_argument(
        "--record-id",
        help="Identifiant spécifique d'enregistrement FASTA si plusieurs séquences sont présentes.",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        help="Fichier CSV d'annotation start,end,name[,product] pour définir les protéines.",
    )
    parser.add_argument(
        "--domain-csv",
        type=Path,
        help="Fichier CSV de domaines (start,end,name[,protein,product]) pour agréger les scores.",
    )
    parser.add_argument(
        "--domain-summary-name",
        default=DEFAULT_DOMAIN_SUMMARY_NAME,
        help=f"Nom du TSV pour le score moyen par domaine (défaut: {DEFAULT_DOMAIN_SUMMARY_NAME}).",
    )
    parser.add_argument(
        "--label",
        default="principal",
        help="Étiquette pour le jeu de données principal (défaut: principal).",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        help=(
            "Entrée de comparaison inter-gènes au format label|log|fasta"
            "[|annotation][|record_id]. Option répétable."
        ),
    )
    parser.add_argument(
        "--intergene-summary-name",
        default=DEFAULT_INTERGENE_SUMMARY_NAME,
        help=f"Nom du TSV de comparaison inter-gènes (défaut: {DEFAULT_INTERGENE_SUMMARY_NAME}).",
    )
    parser.add_argument(
        "--intergene-figure-name",
        default=DEFAULT_INTERGENE_FIGURE_NAME,
        help=f"Nom du graphique de comparaison inter-gènes (défaut: {DEFAULT_INTERGENE_FIGURE_NAME}).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.log.exists():
        parser.error(f"Le fichier de log {args.log} est introuvable.")

    (
        records,
        records_frame,
        counts,
        lengths,
        per_run_frame,
        protein_stats,
        category_stats,
        dataset_summary,
    ) = analyse_dataset(
        log_path=args.log,
        fasta_path=args.fasta,
        record_id=args.record_id,
        annotation_path=args.annotation,
    )

    run_count = 1
    if not per_run_frame.empty:
        detected = per_run_frame["simulation"].nunique()
        if detected > 0:
            run_count = detected

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = merge_counts_lengths(counts, lengths, run_count=run_count)
    counts_path = args.output_dir / args.counts_name
    write_counts_table(rows, counts_path)
    print(f"Table de comptage enregistrée dans {counts_path}")

    tolerance_stats_path = args.output_dir / args.tolerance_stats_name
    write_tolerance_stats_table(protein_stats, tolerance_stats_path)
    print(f"Statistiques détaillées enregistrées dans {tolerance_stats_path}")

    frame = build_dataframe(counts, lengths, run_count=run_count)
    heatmap_path = args.output_dir / args.heatmap_name
    render_heatmap(frame, heatmap_path, dpi=args.dpi, cmap=args.cmap)

    errorbar_path = args.output_dir / args.errorbars_name
    render_tolerance_errorbars(protein_stats, errorbar_path, dpi=args.dpi)

    scatter_path = args.output_dir / args.scatter_name
    render_correlation_scatter(protein_stats, scatter_path, dpi=args.dpi, cmap=args.cmap)

    category_summary_path = args.output_dir / args.category_summary_name
    write_category_summary(category_stats, category_summary_path)
    print(f"Résumé fonctionnel enregistré dans {category_summary_path}")

    domain_definitions = load_domain_definitions(args.domain_csv) if args.domain_csv else {}
    if domain_definitions:
        domain_summary = compute_domain_summary(records, domain_definitions)
        domain_summary_path = args.output_dir / args.domain_summary_name
        write_domain_summary(domain_summary, domain_summary_path)
        print(f"Résumé par domaine enregistré dans {domain_summary_path}")
    else:
        print("Aucun domaine fonctionnel fourni; saut de l'agrégation par domaine.")

    try:
        density_matrices = compute_mutation_density_matrices(
            records,
            lengths,
            bin_size=args.density_bin,
        )
    except ValueError as exc:
        parser.error(str(exc))
    density_path = args.output_dir / args.density_name
    render_mutation_density_heatmaps(
        density_matrices,
        density_path,
        dpi=args.dpi,
        cmap=args.density_cmap or args.cmap,
    )

    dataset_summary_frame = pd.DataFrame([dataset_summary], index=[0])
    dataset_summary_frame["label"] = args.label

    comparison_rows: list[pd.Series] = []
    for comparison_arg in args.comparison:
        try:
            label, log_path, fasta_path, annotation_path, comparison_record_id = parse_comparison_argument(
                comparison_arg
            )
        except ValueError as exc:
            parser.error(str(exc))
        if not log_path.exists():
            parser.error(f"Le log de comparaison '{log_path}' est introuvable.")
        if not fasta_path.exists():
            parser.error(f"Le FASTA de comparaison '{fasta_path}' est introuvable.")
        (
            _,
            comparison_records_frame,
            _,
            comparison_lengths,
            comparison_per_run,
            _,
            _,
            comparison_summary,
        ) = analyse_dataset(
            log_path=log_path,
            fasta_path=fasta_path,
            record_id=comparison_record_id,
            annotation_path=annotation_path,
        )
        comparison_summary = comparison_summary.copy()
        comparison_summary["label"] = label
        comparison_rows.append(comparison_summary)

    intergene_summary = pd.concat(
        [dataset_summary_frame] + [row.to_frame().T for row in comparison_rows],
        ignore_index=True,
    )
    intergene_summary = intergene_summary[
        [
            "label",
            "runs",
            "mean_tolerance",
            "std_tolerance",
            "ci95_tolerance",
            "accepted_total",
            "total_length_aa",
            "mean_blosum",
            "mean_acceptance_probability",
            "mean_grantham",
            "mean_hydrophobicity_delta",
        ]
    ]

    intergene_summary_path = args.output_dir / args.intergene_summary_name
    write_intergene_summary(intergene_summary, intergene_summary_path)
    print(f"Résumé inter-gènes enregistré dans {intergene_summary_path}")

    intergene_figure_path = args.output_dir / args.intergene_figure_name
    render_intergene_comparison_plot(intergene_summary, intergene_figure_path, dpi=args.dpi)


if __name__ == "__main__":
    main()
