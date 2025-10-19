"""Simulate random nucleotide mutations and classify their protein-level effects."""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

from Bio import SeqIO
from Bio.Align import substitution_matrices
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

NUCLEOTIDES: tuple[str, ...] = ("A", "T", "G", "C")
TRANSITION_PARTNERS: dict[str, tuple[str, ...]] = {
    "A": ("G",),
    "G": ("A",),
    "C": ("T",),
    "T": ("C",),
}


@dataclass(slots=True, frozen=True)
class MutationOutcome:
    position: int
    original_base: str
    mutated_base: str
    codon_index: int
    original_codon: str
    mutated_codon: str
    original_amino_acid: str
    mutated_amino_acid: str
    blosum_score: float
    acceptance_probability: float
    codon_preference: float
    severity_score: int
    grantham_distance: float
    hydrophobicity_delta: float
    flags: tuple[str, ...]
    accepted: bool

    @property
    def is_silent(self) -> bool:
        return self.original_amino_acid == self.mutated_amino_acid


@dataclass(slots=True, frozen=True)
class GenomeFeature:
    start: int  # 1-based inclusive
    end: int  # 1-based inclusive
    name: str
    product: str


@dataclass(slots=True, frozen=True)
class IndelOutcome:
    position: int
    length_nt: int
    kind: Literal["insertion", "deletion"]
    sequence: str
    feature: GenomeFeature | None
    accepted: bool


@dataclass(slots=True, frozen=True)
class SelectionModel:
    selection_strength: float
    codon_usage_weight: float
    codon_preferences: dict[str, float]

    def acceptance_factors(
        self,
        *,
        original_codon: str,
        mutated_codon: str,
        original_amino_acid: str,
        mutated_amino_acid: str,
    ) -> tuple[float, float]:
        """Return selection factor and codon preference for the proposed mutation."""

        codon_preference = self.codon_preferences.get(mutated_codon, CODON_PREFERENCE_FLOOR)
        codon_preference = max(codon_preference, CODON_PREFERENCE_FLOOR)
        codon_term = 0.0
        if self.codon_usage_weight > 0:
            codon_term = self.codon_usage_weight * (codon_preference - 0.5) * CODON_USAGE_LOGIT_SCALE

        if mutated_amino_acid == "*":
            return 0.0, codon_preference

        if mutated_amino_acid == original_amino_acid:
            return 1.0, codon_preference

        selection_term = 0.0
        if self.selection_strength > 0:
            score = blosum_score(original_amino_acid, mutated_amino_acid)
            selection_term = self.selection_strength * score

        probability = sigmoid(selection_term + codon_term)
        return probability, codon_preference


BLOSUM62_MATRIX = substitution_matrices.load("BLOSUM62")
CODON_PREFERENCE_FLOOR = 0.05
CODON_USAGE_LOGIT_SCALE = 4.0
MAX_ACCEPTANCE_ATTEMPTS = 100


def sigmoid(value: float) -> float:
    """Return a numerically stable logistic transform."""

    if value >= 60:
        return 1.0
    if value <= -60:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))

HYDROPHOBICITY_KD: dict[str, float] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}
GRANTHAM_DISTANCE: dict[tuple[str, str], int] = {
    ("A", "R"): 112,
    ("A", "N"): 111,
    ("A", "D"): 126,
    ("A", "C"): 195,
    ("A", "Q"): 91,
    ("A", "E"): 107,
    ("A", "G"): 60,
    ("A", "H"): 86,
    ("A", "I"): 94,
    ("A", "L"): 96,
    ("A", "K"): 106,
    ("A", "M"): 84,
    ("A", "F"): 113,
    ("A", "P"): 27,
    ("A", "S"): 99,
    ("A", "T"): 58,
    ("A", "W"): 148,
    ("A", "Y"): 112,
    ("A", "V"): 64,
    ("R", "N"): 86,
    ("R", "D"): 96,
    ("R", "C"): 180,
    ("R", "Q"): 43,
    ("R", "E"): 54,
    ("R", "G"): 125,
    ("R", "H"): 29,
    ("R", "I"): 97,
    ("R", "L"): 102,
    ("R", "K"): 26,
    ("R", "M"): 91,
    ("R", "F"): 97,
    ("R", "P"): 103,
    ("R", "S"): 110,
    ("R", "T"): 71,
    ("R", "W"): 101,
    ("R", "Y"): 77,
    ("R", "V"): 96,
    ("N", "D"): 23,
    ("N", "C"): 139,
    ("N", "Q"): 46,
    ("N", "E"): 42,
    ("N", "G"): 80,
    ("N", "H"): 68,
    ("N", "I"): 149,
    ("N", "L"): 153,
    ("N", "K"): 94,
    ("N", "M"): 142,
    ("N", "F"): 158,
    ("N", "P"): 91,
    ("N", "S"): 46,
    ("N", "T"): 65,
    ("N", "W"): 174,
    ("N", "Y"): 143,
    ("N", "V"): 133,
    ("D", "C"): 154,
    ("D", "Q"): 61,
    ("D", "E"): 45,
    ("D", "G"): 94,
    ("D", "H"): 81,
    ("D", "I"): 168,
    ("D", "L"): 172,
    ("D", "K"): 101,
    ("D", "M"): 160,
    ("D", "F"): 177,
    ("D", "P"): 108,
    ("D", "S"): 65,
    ("D", "T"): 85,
    ("D", "W"): 181,
    ("D", "Y"): 160,
    ("D", "V"): 152,
    ("C", "Q"): 154,
    ("C", "E"): 170,
    ("C", "G"): 159,
    ("C", "H"): 174,
    ("C", "I"): 198,
    ("C", "L"): 198,
    ("C", "K"): 202,
    ("C", "M"): 196,
    ("C", "F"): 205,
    ("C", "P"): 169,
    ("C", "S"): 112,
    ("C", "T"): 149,
    ("C", "W"): 215,
    ("C", "Y"): 194,
    ("C", "V"): 192,
    ("Q", "E"): 29,
    ("Q", "G"): 87,
    ("Q", "H"): 24,
    ("Q", "I"): 109,
    ("Q", "L"): 113,
    ("Q", "K"): 53,
    ("Q", "M"): 101,
    ("Q", "F"): 116,
    ("Q", "P"): 76,
    ("Q", "S"): 68,
    ("Q", "T"): 42,
    ("Q", "W"): 130,
    ("Q", "Y"): 99,
    ("Q", "V"): 96,
    ("E", "G"): 98,
    ("E", "H"): 40,
    ("E", "I"): 134,
    ("E", "L"): 138,
    ("E", "K"): 56,
    ("E", "M"): 126,
    ("E", "F"): 140,
    ("E", "P"): 93,
    ("E", "S"): 80,
    ("E", "T"): 65,
    ("E", "W"): 152,
    ("E", "Y"): 122,
    ("E", "V"): 121,
    ("G", "H"): 94,
    ("G", "I"): 135,
    ("G", "L"): 138,
    ("G", "K"): 127,
    ("G", "M"): 127,
    ("G", "F"): 153,
    ("G", "P"): 42,
    ("G", "S"): 56,
    ("G", "T"): 59,
    ("G", "W"): 184,
    ("G", "Y"): 147,
    ("G", "V"): 109,
    ("H", "I"): 94,
    ("H", "L"): 99,
    ("H", "K"): 32,
    ("H", "M"): 87,
    ("H", "F"): 100,
    ("H", "P"): 77,
    ("H", "S"): 89,
    ("H", "T"): 47,
    ("H", "W"): 115,
    ("H", "Y"): 83,
    ("H", "V"): 84,
    ("I", "L"): 5,
    ("I", "K"): 102,
    ("I", "M"): 10,
    ("I", "F"): 21,
    ("I", "P"): 95,
    ("I", "S"): 142,
    ("I", "T"): 89,
    ("I", "W"): 61,
    ("I", "Y"): 33,
    ("I", "V"): 29,
    ("L", "K"): 107,
    ("L", "M"): 15,
    ("L", "F"): 22,
    ("L", "P"): 98,
    ("L", "S"): 145,
    ("L", "T"): 92,
    ("L", "W"): 61,
    ("L", "Y"): 36,
    ("L", "V"): 32,
    ("K", "M"): 95,
    ("K", "F"): 102,
    ("K", "P"): 103,
    ("K", "S"): 121,
    ("K", "T"): 78,
    ("K", "W"): 110,
    ("K", "Y"): 85,
    ("K", "V"): 97,
    ("M", "F"): 28,
    ("M", "P"): 87,
    ("M", "S"): 135,
    ("M", "T"): 81,
    ("M", "W"): 67,
    ("M", "Y"): 36,
    ("M", "V"): 21,
    ("F", "P"): 114,
    ("F", "S"): 155,
    ("F", "T"): 103,
    ("F", "W"): 40,
    ("F", "Y"): 22,
    ("F", "V"): 50,
    ("P", "S"): 74,
    ("P", "T"): 38,
    ("P", "W"): 147,
    ("P", "Y"): 110,
    ("P", "V"): 68,
    ("S", "T"): 58,
    ("S", "W"): 177,
    ("S", "Y"): 144,
    ("S", "V"): 124,
    ("T", "W"): 128,
    ("T", "Y"): 92,
    ("T", "V"): 69,
    ("W", "Y"): 37,
    ("W", "V"): 88,
    ("Y", "V"): 55,
}
AMINO_CLASSES: dict[str, str] = {
    "A": "nonpolar",
    "V": "nonpolar",
    "L": "nonpolar",
    "I": "nonpolar",
    "M": "nonpolar",
    "F": "aromatic",
    "Y": "aromatic",
    "W": "aromatic",
    "P": "special",
    "G": "special",
    "S": "polar",
    "T": "polar",
    "C": "special",
    "N": "polar",
    "Q": "polar",
    "K": "positive",
    "R": "positive",
    "H": "positive",
    "D": "negative",
    "E": "negative",
}


def blosum_score(original: str, mutated: str) -> float:
    if original == mutated:
        return 0.0

    try:
        return float(BLOSUM62_MATRIX[(original, mutated)])
    except KeyError:
        try:
            return float(BLOSUM62_MATRIX[(mutated, original)])
        except KeyError:
            return -4.0


def amino_acid_class(amino: str) -> str:
    return AMINO_CLASSES.get(amino, "unknown")


def hydrophobicity(amino: str) -> float:
    return HYDROPHOBICITY_KD.get(amino, 0.0)


def grantham_distance(original: str, mutated: str) -> float:
    if original == mutated:
        return 0.0
    pair = (original, mutated)
    if pair in GRANTHAM_DISTANCE:
        return float(GRANTHAM_DISTANCE[pair])
    pair = (mutated, original)
    return float(GRANTHAM_DISTANCE.get(pair, 0))


def compute_severity_score(original_amino: str, mutated_amino: str) -> int:
    if original_amino == mutated_amino:
        return -1
    if original_amino == "*" and mutated_amino != "*":
        return 2
    if mutated_amino == "*" and original_amino != "*":
        return 2

    severity = 0
    if amino_acid_class(original_amino) != amino_acid_class(mutated_amino):
        severity += 1
    if original_amino == "P" and mutated_amino != "P":
        severity += 1
    if original_amino == "C" and mutated_amino != "C":
        severity += 1

    return severity


def severity_label(severity: int, *, is_silent: bool, mutated_amino: str, original_amino: str) -> str:
    if is_silent:
        return "synonyme"
    if severity >= 2 or original_amino == "*" or mutated_amino == "*":
        return "critique"
    if severity == 1:
        return "drastique"
    return "conservateur"


def compute_codon_preferences(seq: Seq) -> dict[str, float]:
    """Return codon preference scores normalisés par acide aminé."""

    codon_counts: Counter[str] = Counter()
    seq_str = str(seq)
    for index in range(0, len(seq_str), 3):
        codon = seq_str[index : index + 3]
        if len(codon) == 3:
            codon_counts[codon] += 1

    amino_to_counts: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
    for codon, count in codon_counts.items():
        amino = str(Seq(codon).translate())
        amino_to_counts[amino].append((codon, count))

    preferences: dict[str, float] = {}
    for amino, codons in amino_to_counts.items():
        max_count = max(count for _, count in codons)
        if max_count == 0:
            continue
        for codon, count in codons:
            preferences[codon] = count / max_count

    return preferences


def build_selection_model(
    seq: Seq,
    selection_strength: float,
    codon_usage_weight: float,
) -> SelectionModel | None:
    if selection_strength <= 0 and codon_usage_weight <= 0:
        return None

    preferences = compute_codon_preferences(seq)
    return SelectionModel(
        selection_strength=selection_strength,
        codon_usage_weight=codon_usage_weight,
        codon_preferences=preferences,
    )


def build_codon_library(seq: Seq) -> list[str]:
    """Return the list of codons observed in the référence."""

    seq_str = str(seq)
    return [seq_str[index : index + 3] for index in range(0, len(seq_str), 3) if len(seq_str[index : index + 3]) == 3]


def simulate_indel_event(
    seq: Seq,
    rng: random.Random,
    *,
    probability: float,
    max_codon_length: int,
    proofreading_acceptance: float,
    codon_library: Sequence[str],
    feature_lookup: Sequence[GenomeFeature | None],
) -> tuple[Seq, list[IndelOutcome]]:
    """Optionally introduce a rare in-frame indel filtered by proofreading."""

    if probability <= 0 or rng.random() >= probability:
        return seq, []

    if max_codon_length <= 0:
        max_codon_length = 1

    seq_str = str(seq)
    codon_count = len(seq_str) // 3
    outcomes: list[IndelOutcome] = []

    # Determine indel type; fallback to insertion if deletion is impossible.
    choose_deletion = rng.random() < 0.5 and codon_count > 1
    if choose_deletion and codon_count <= 1:
        choose_deletion = False

    if choose_deletion:
        max_length = min(max_codon_length, codon_count - 1)
        if max_length <= 0:
            return seq, []
        length_codons = rng.randint(1, max_length)
        start_codon = rng.randrange(0, codon_count - length_codons + 1)
        start_index = start_codon * 3
        end_index = start_index + length_codons * 3
        deleted = seq_str[start_index:end_index]
        feature = feature_lookup[start_index] if feature_lookup and start_index < len(feature_lookup) else None

        accepted = rng.random() < proofreading_acceptance
        if accepted:
            seq_str = seq_str[:start_index] + seq_str[end_index:]
        outcomes.append(
            IndelOutcome(
                position=start_index,
                length_nt=len(deleted),
                kind="deletion",
                sequence=deleted,
                feature=feature,
                accepted=accepted,
            )
        )
        return Seq(seq_str), outcomes

    # Insertion branch
    insert_position_codon = rng.randrange(0, codon_count + 1)
    insert_index = insert_position_codon * 3
    length_codons = rng.randint(1, max_codon_length)

    if codon_library:
        codon_counter = Counter(codon_library)
        codon_options = list(codon_counter.keys())
        codon_weights = [max(0.01, float(count)) for count in codon_counter.values()]
    else:
        codon_options = ["ATG"]
        codon_weights = [1.0]

    inserted_codons = [
        weighted_choice(codon_options, codon_weights, rng)
        for _ in range(length_codons)
    ]
    inserted_seq = "".join(inserted_codons)

    accepted = rng.random() < proofreading_acceptance
    if accepted:
        seq_str = seq_str[:insert_index] + inserted_seq + seq_str[insert_index:]

    feature_index = min(insert_index, len(feature_lookup) - 1) if feature_lookup else 0
    feature = feature_lookup[feature_index] if feature_lookup and 0 <= feature_index < len(feature_lookup) else None

    outcomes.append(
        IndelOutcome(
            position=insert_index,
            length_nt=len(inserted_seq),
            kind="insertion",
            sequence=inserted_seq,
            feature=feature,
            accepted=accepted,
        )
    )

    return Seq(seq_str), outcomes


def weighted_choice(options: Sequence[str], weights: Sequence[float], rng: random.Random) -> str:
    """Return a single option sampled according to the provided weights."""

    if len(options) != len(weights):
        raise ValueError("Le nombre de poids doit correspondre au nombre d'options.")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Les poids doivent contenir au moins une valeur positive.")

    threshold = rng.random() * total_weight
    cumulative = 0.0
    for option, weight in zip(options, weights):
        cumulative += weight
        if threshold <= cumulative:
            return option
    return options[-1]


def weighted_sample_without_replacement(
    population: Sequence[int],
    weights: Sequence[float],
    count: int,
    rng: random.Random,
) -> list[int]:
    """Sample indices without replacement using weighted probabilities."""

    if count <= 0:
        return []
    if len(population) != len(weights):
        raise ValueError("La population et les poids doivent avoir la même taille.")
    if count > len(population):
        raise ValueError("Impossible de sélectionner plus d'éléments que la population disponible.")

    remaining_indices = list(range(len(population)))
    remaining_weights = list(weights)
    result: list[int] = []

    for _ in range(count):
        total_weight = sum(remaining_weights)
        if total_weight <= 0:
            raise ValueError("Les poids restants doivent contenir au moins une valeur positive.")
        threshold = rng.random() * total_weight
        cumulative = 0.0
        for rel_index, weight in enumerate(remaining_weights):
            cumulative += weight
            if threshold <= cumulative:
                population_index = remaining_indices.pop(rel_index)
                result.append(population[population_index])
                remaining_weights.pop(rel_index)
                break

    return result


def load_record(fasta_path: Path, record_id: str | None) -> SeqRecord:
    """Return the requested sequence record or fallback to the first one."""
    with fasta_path.open() as handle:
        for record in SeqIO.parse(handle, "fasta"):
            if record_id is None:
                return record
            if record.id == record_id or record.description.startswith(record_id):
                return record
    raise ValueError(f"Aucune séquence trouvée pour l'identifiant '{record_id}'.")


def prepare_coding_sequence(seq: Seq) -> Seq:
    """Trim the sequence to a length divisible by three for translation."""
    normalized = seq.upper()
    trimmed_length = len(normalized) - (len(normalized) % 3)
    if trimmed_length == 0:
        raise ValueError("La séquence est trop courte pour être traduite.")
    return normalized[:trimmed_length]


def choose_mutation_positions(
    seq: Seq,
    count: int,
    rng: random.Random,
    position_weights: Sequence[float] | None = None,
) -> list[int]:
    """Pick nucleotide positions eligible for mutation."""

    seq_str = str(seq)
    if position_weights is not None and len(position_weights) != len(seq_str):
        raise ValueError("Le nombre de poids de positions doit correspondre à la longueur de la séquence.")

    candidates: list[int] = []
    weights: list[float] = []

    for i, base in enumerate(seq_str):
        if base not in NUCLEOTIDES:
            continue
        weight = position_weights[i] if position_weights is not None else 1.0
        if weight <= 0:
            continue
        candidates.append(i)
        weights.append(weight)

    if len(candidates) < count:
        raise ValueError("Pas assez de nucléotides A/T/G/C pour appliquer les mutations demandées.")

    if position_weights is None:
        return rng.sample(candidates, count)

    return weighted_sample_without_replacement(candidates, weights, count, rng)


def mutate_sequence(
    seq: Seq,
    positions: Sequence[int],
    rng: random.Random,
    *,
    transition_weight: float = 2.0,
    transversion_weight: float = 1.0,
    selection_model: SelectionModel | None = None,
) -> tuple[Seq, list[MutationOutcome]]:
    """Apply point mutations and capture their codon-level impact."""
    original_chars = list(str(seq))
    mutated_chars = original_chars.copy()
    outcomes: list[MutationOutcome] = []

    for pos in sorted(positions):
        original_base = original_chars[pos]
        transitions = TRANSITION_PARTNERS.get(original_base)
        if transitions is None:
            raise ValueError(f"Base inconnue pour la mutation: {original_base}")

        options: list[str] = []
        weights: list[float] = []
        candidate_data: list[tuple[str, str, str, float, float]] = []
        codon_index = pos // 3
        codon_start = codon_index * 3
        original_codon = "".join(original_chars[codon_start:codon_start + 3])
        original_amino_acid = str(Seq(original_codon).translate())

        for candidate_base in NUCLEOTIDES:
            if candidate_base == original_base:
                continue
            ts_tv_weight = transition_weight if transitions and candidate_base in transitions else transversion_weight
            if ts_tv_weight <= 0:
                continue

            mutated_chars[pos] = candidate_base
            mutated_codon = "".join(mutated_chars[codon_start:codon_start + 3])
            mutated_amino_acid = str(Seq(mutated_codon).translate())

            if selection_model:
                acceptance_probability, codon_preference = selection_model.acceptance_factors(
                    original_codon=original_codon,
                    mutated_codon=mutated_codon,
                    original_amino_acid=original_amino_acid,
                    mutated_amino_acid=mutated_amino_acid,
                )
            else:
                acceptance_probability = 1.0
                codon_preference = 1.0

            options.append(candidate_base)
            weights.append(ts_tv_weight)
            candidate_data.append(
                (
                    candidate_base,
                    mutated_codon,
                    mutated_amino_acid,
                    acceptance_probability,
                    codon_preference,
                )
            )

            mutated_chars[pos] = original_base

        if not candidate_data:
            raise ValueError(
                "Aucune mutation admissible avec les paramètres fournis: assouplissez la sélection ou les poids Ts/Tv."
            )

        index_lookup = {base: idx for idx, base in enumerate(options)}

        def build_outcome(
            candidate_entry: tuple[str, str, str, float, float],
            accepted: bool,
            extra_flags: tuple[str, ...] = (),
        ) -> MutationOutcome:
            candidate_base, mutated_codon, mutated_amino_acid, acceptance_probability, codon_preference = candidate_entry
            blosum = blosum_score(original_amino_acid, mutated_amino_acid)
            severity = compute_severity_score(original_amino_acid, mutated_amino_acid)
            grantham = grantham_distance(original_amino_acid, mutated_amino_acid)
            hydro_delta = hydrophobicity(mutated_amino_acid) - hydrophobicity(original_amino_acid)
            flag_buffer: list[str] = []
            if original_amino_acid == "*" and mutated_amino_acid != "*":
                flag_buffer.append("stop_perdu")
            if mutated_amino_acid == "*" and original_amino_acid != "*":
                flag_buffer.append("stop_gagné")
            if original_amino_acid == "P" and mutated_amino_acid != "P":
                flag_buffer.append("proline_perdue")
            if original_amino_acid == "C" and mutated_amino_acid != "C":
                flag_buffer.append("cysteine_perdue")
            if original_amino_acid == "G" and mutated_amino_acid != "G":
                flag_buffer.append("glycine_perdue")
            flag_buffer.extend(extra_flags)

            return MutationOutcome(
                position=pos,
                original_base=original_base,
                mutated_base=candidate_base,
                codon_index=codon_index,
                original_codon=original_codon,
                mutated_codon=mutated_codon,
                original_amino_acid=original_amino_acid,
                mutated_amino_acid=mutated_amino_acid,
                blosum_score=blosum,
                acceptance_probability=acceptance_probability,
                codon_preference=codon_preference,
                severity_score=severity,
                grantham_distance=grantham,
                hydrophobicity_delta=hydro_delta,
                flags=tuple(flag_buffer),
                accepted=accepted,
            )

        attempts = 0
        while True:
            attempts += 1
            mutated_base = weighted_choice(options, weights, rng)
            candidate_index = index_lookup[mutated_base]
            candidate_entry = candidate_data[candidate_index]
            acceptance_probability = candidate_entry[3]
            accepted = rng.random() < acceptance_probability

            if accepted:
                mutated_chars[pos] = mutated_base
                outcomes.append(build_outcome(candidate_entry, True))
                break

            mutated_chars[pos] = original_base
            outcomes.append(build_outcome(candidate_entry, False))

            if attempts >= MAX_ACCEPTANCE_ATTEMPTS:
                best_index = max(range(len(candidate_data)), key=lambda idx: candidate_data[idx][3])
                best_candidate = candidate_data[best_index]
                mutated_chars[pos] = best_candidate[0]
                outcomes.append(build_outcome(best_candidate, True, extra_flags=("accept_forcé",)))
                break

    mutated_seq = Seq("".join(mutated_chars))
    return mutated_seq, outcomes


def load_features(annotation_path: Path | None, record: SeqRecord) -> list[GenomeFeature]:
    """Return genome features from a manual file or default heuristics."""

    if annotation_path:
        with annotation_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"start", "end", "name"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    "Le fichier d'annotation doit contenir les colonnes start,end,name (et optionnellement product)."
                )
            features = [
                GenomeFeature(
                    start=int(row["start"]),
                    end=int(row["end"]),
                    name=row["name"].strip(),
                    product=(row.get("product") or row["name"]).strip(),
                )
                for row in reader
            ]
            if not features:
                raise ValueError("Aucune annotation chargée: vérifiez le contenu du fichier.")
            return features

    text = f"{record.id} {record.description}".lower()
    if "sars" in text or "severe acute respiratory syndrome coronavirus 2" in text:
        # NC_045512.2 reference coordinates (1-based inclusive)
        return [
            GenomeFeature(266, 805, "nsp1", "Leader protein"),
            GenomeFeature(806, 2719, "nsp2", "Non-structural protein 2"),
            GenomeFeature(2720, 8554, "nsp3", "Papain-like proteinase"),
            GenomeFeature(8555, 10054, "nsp4", "Hydrophobic protein"),
            GenomeFeature(10055, 10972, "nsp5", "3C-like proteinase"),
            GenomeFeature(10973, 11842, "nsp6", "Non-structural protein 6"),
            GenomeFeature(11843, 12091, "nsp7", "Non-structural protein 7"),
            GenomeFeature(12092, 12685, "nsp8", "Non-structural protein 8"),
            GenomeFeature(12686, 13024, "nsp9", "Non-structural protein 9"),
            GenomeFeature(13025, 13441, "nsp10", "Non-structural protein 10"),
            GenomeFeature(13442, 16236, "nsp12", "RNA-dependent RNA polymerase"),
            GenomeFeature(16237, 18039, "nsp13", "Helicase"),
            GenomeFeature(18040, 19620, "nsp14", "Exoribonuclease"),
            GenomeFeature(19621, 20658, "nsp15", "Endoribonuclease"),
            GenomeFeature(20659, 21552, "nsp16", "2'-O-methyltransferase"),
            GenomeFeature(21563, 25384, "S", "Spike glycoprotein"),
            GenomeFeature(25393, 26220, "ORF3a", "Protein 3a"),
            GenomeFeature(26245, 26472, "E", "Envelope small membrane protein"),
            GenomeFeature(26523, 27191, "M", "Membrane glycoprotein"),
            GenomeFeature(27202, 27387, "ORF6", "Protein 6"),
            GenomeFeature(27394, 27759, "ORF7a", "Protein 7a"),
            GenomeFeature(27756, 27887, "ORF7b", "Protein 7b"),
            GenomeFeature(27894, 28259, "ORF8", "Protein 8"),
            GenomeFeature(28274, 29533, "N", "Nucleocapsid phosphoprotein"),
            GenomeFeature(29558, 29674, "ORF10", "Protein 10"),
        ]

    return []


def feature_for_position(features: Iterable[GenomeFeature], nucleotide_position: int) -> GenomeFeature | None:
    """Return the genome feature that overlaps the given 1-based nucleotide position."""
    for feature in features:
        if feature.start <= nucleotide_position <= feature.end:
            return feature
    return None


def describe_feature(feature: GenomeFeature | None) -> str:
    if feature is None:
        return "région non annotée"
    if feature.product and feature.product != feature.name:
        return f"{feature.product} ({feature.name})"
    return feature.name


def sanitize_identifier(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text)


def build_feature_lookup(features: Sequence[GenomeFeature], seq_length: int) -> list[GenomeFeature | None]:
    """Return a positional lookup for constant-time feature access."""

    lookup: list[GenomeFeature | None] = [None] * seq_length
    for feature in features:
        start_index = max(feature.start - 1, 0)
        end_index = min(feature.end, seq_length)
        for index in range(start_index, end_index):
            lookup[index] = feature
    return lookup


def compute_feature_codon_lengths(feature_lookup: Sequence[GenomeFeature | None]) -> dict[str, int]:
    """Return the number of codons assigned to each annotated feature."""

    lengths: defaultdict[str, int] = defaultdict(int)
    codon_count = len(feature_lookup) // 3
    for codon_index in range(codon_count):
        start = codon_index * 3
        slice_features = feature_lookup[start : start + 3]
        feature_counts: Counter[GenomeFeature] = Counter(
            candidate for candidate in slice_features if candidate is not None
        )
        feature = feature_counts.most_common(1)[0][0] if feature_counts else None
        feature_key = feature.name if feature else "non_annotée"
        lengths[feature_key] += 1
    return dict(lengths)


def compute_position_weights(
    seq: Seq,
    feature_lookup: Sequence[GenomeFeature | None],
    cpg_weight: float,
    feature_weight_overrides: dict[str, float],
) -> list[float]:
    """Compute per-position weights combining hotspots and feature-specific μ."""

    seq_str = str(seq)
    weights = [1.0] * len(seq_str)

    if cpg_weight != 1.0:
        for index, base in enumerate(seq_str):
            if base == "C" and index + 1 < len(seq_str) and seq_str[index + 1] == "G":
                weights[index] *= cpg_weight
            if base == "G" and index > 0 and seq_str[index - 1] == "C":
                weights[index] *= cpg_weight

    if feature_weight_overrides:
        for index, feature in enumerate(feature_lookup):
            if feature is None:
                continue
            multiplier = feature_weight_overrides.get(feature.name)
            if multiplier is None and feature.product:
                multiplier = feature_weight_overrides.get(feature.product)
            if multiplier is not None:
                weights[index] *= multiplier

    return weights


def parse_feature_weight_entries(entries: Sequence[str]) -> dict[str, float]:
    """Parse CLI overrides formatted as name=value for feature weights."""

    overrides: dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Entrée de pondération invalide: '{entry}' (format attendu: nom=valeur)")
        name, value = entry.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("Le nom de caractéristique ne peut pas être vide.")
        try:
            overrides[name] = float(value)
        except ValueError as exc:  # pragma: no cover - message utilisateur
            raise ValueError(f"Valeur numérique invalide pour '{name}': '{value}'") from exc
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simuler des mutations aléatoires sur une séquence ADN.")
    parser.add_argument(
        "--fasta",
        type=Path,
        default=Path("Start/sequences.fasta"),
        help="Chemin vers le fichier FASTA (défaut: Start/sequences.fasta).",
    )
    parser.add_argument(
        "--record-id",
        help="Identifiant exact ou préfixe de la séquence à muter (défaut: première séquence).",
    )
    parser.add_argument(
        "--mutations",
        type=int,
        default=1,
        help="Nombre de mutations ponctuelles à appliquer par simulation (défaut: 1).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Nombre de simulations indépendantes à exécuter (défaut: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Graine aléatoire pour reproduire les simulations.",
    )
    parser.add_argument(
        "--protein-fasta",
        type=Path,
        help="Chemin de sortie pour enregistrer les protéines (FASTA).",
    )
    parser.add_argument(
        "--mutations-csv",
        type=Path,
        help="Chemin de sortie pour enregistrer les mutations acceptées (CSV).",
    )
    parser.add_argument(
        "--preview-aa",
        type=int,
        default=60,
        help="Nombre d'acides aminés à afficher pour chaque protéine (0 = rien).",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        help="Fichier CSV avec les colonnes start,end,name[,product] pour nommer les protéines.",
    )
    parser.add_argument(
        "--transition-weight",
        type=float,
        default=2.0,
        help="Poids relatif des transitions (défaut: 2.0).",
    )
    parser.add_argument(
        "--transversion-weight",
        type=float,
        default=1.0,
        help="Poids relatif des transversions (défaut: 1.0).",
    )
    parser.add_argument(
        "--hotspot-cpg-weight",
        type=float,
        default=1.0,
        help="Facteur multiplicatif pour les positions CpG (défaut: 1.0).",
    )
    parser.add_argument(
        "--feature-weight",
        action="append",
        default=[],
        metavar="IDENTIFIANT=MULTIPLICATEUR",
        help="Ajuste la probabilité de mutation pour une protéine (nom ou produit).",
    )
    parser.add_argument(
        "--selection-strength",
        type=float,
        default=1.0,
        help="Intensité de sélection (BLOSUM) appliquée aux mutations (défaut: 1.0).",
    )
    parser.add_argument(
        "--codon-usage-weight",
        type=float,
        default=0.5,
        help="Poids du biais d'usage des codons dans l'acceptation (défaut: 0.5).",
    )
    parser.add_argument(
        "--indel-probability",
        type=float,
        default=0.0,
        help="Probabilité d'introduire un indel in-frame par simulation (défaut: 0).",
    )
    parser.add_argument(
        "--indel-max-codons",
        type=int,
        default=1,
        help="Longueur maximale (en codons) d'une insertion ou suppression (défaut: 1).",
    )
    parser.add_argument(
        "--indel-proofreading",
        type=float,
        default=0.1,
        help="Probabilité qu'un indel échappe au proofreading et soit conservé (défaut: 0.1).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sequence_record = load_record(args.fasta, args.record_id)
    coding_seq = prepare_coding_sequence(sequence_record.seq)

    original_protein = coding_seq.translate()
    protein_records: list[SeqRecord] = []
    features = load_features(args.annotation, sequence_record)
    feature_lookup = build_feature_lookup(features, len(coding_seq))
    feature_lengths = compute_feature_codon_lengths(feature_lookup)
    if args.selection_strength < 0:
        parser.error("--selection-strength doit être un nombre positif ou nul.")
    if args.codon_usage_weight < 0:
        parser.error("--codon-usage-weight doit être un nombre positif ou nul.")
    if not 0 <= args.indel_probability <= 1:
        parser.error("--indel-probability doit être compris entre 0 et 1.")
    if args.indel_max_codons <= 0:
        parser.error("--indel-max-codons doit être strictement positif.")
    if not 0 <= args.indel_proofreading <= 1:
        parser.error("--indel-proofreading doit être compris entre 0 et 1.")

    try:
        feature_weight_overrides = parse_feature_weight_entries(args.feature_weight)
    except ValueError as exc:
        parser.error(str(exc))
    position_weights = compute_position_weights(
        coding_seq,
        feature_lookup,
        args.hotspot_cpg_weight,
        feature_weight_overrides,
    )
    selection_model = build_selection_model(
        coding_seq,
        args.selection_strength,
        args.codon_usage_weight,
    )
    codon_library = build_codon_library(coding_seq)

    reference_name = sequence_record.description or sequence_record.id
    reference_record = SeqRecord(
        original_protein,
        id=f"{sequence_record.id}_reference",
        description=reference_name,
    )
    protein_records.append(reference_record)

    feature_stats: defaultdict[str, dict[str, float]] = defaultdict(
        lambda: {
            "total": 0,
            "syn": 0,
            "nonsyn": 0,
            "rejected": 0,
            "severity_sum": 0.0,
            "p_accept_sum": 0.0,
            "blosum_sum": 0.0,
            "grantham_sum": 0.0,
            "hydro_sum": 0.0,
            "danger": 0,
            "indels": 0,
        }
    )
    feature_display_names: dict[str, str] = {"non_annotée": describe_feature(None)}
    mutation_rows: list[str] = []
    rejected_rows: list[str] = []
    red_flags: list[str] = []
    indel_records: list[str] = []
    mutations_export_rows: list[dict[str, object]] = []
    accepted_blosum_scores: list[float] = []
    rejected_blosum_scores: list[float] = []
    accepted_probabilities: list[float] = []
    rejected_probabilities: list[float] = []
    grantham_values: list[float] = []
    hydrophobicity_deltas: list[float] = []
    total_synonymous = 0
    total_nonsynonymous = 0
    total_rejected = 0

    print(f"Séquence chargée: {args.fasta} ({len(coding_seq)} nt traduisibles)")
    print(f"Nom de la protéine de référence: {reference_name}")
    print(f"Longueur: {len(original_protein)} aa")
    if features:
        print(
            "Annotation chargée: "
            + ", ".join(f"{f.product} ({f.name})" for f in features)
        )
    else:
        print("Aucune annotation spécifique: les mutations seront indiquées sans nom connu.")
    if args.preview_aa:
        preview = str(original_protein)[: args.preview_aa]
        suffix = "..." if len(original_protein) > args.preview_aa else ""
        print(f"Aperçu: {preview}{suffix}")
    print("-")

    for sample_index in range(1, args.samples + 1):
        positions = choose_mutation_positions(
            coding_seq,
            args.mutations,
            rng,
            position_weights=position_weights,
        )
        mutated_seq, outcomes = mutate_sequence(
            coding_seq,
            positions,
            rng,
            transition_weight=args.transition_weight,
            transversion_weight=args.transversion_weight,
            selection_model=selection_model,
        )
        mutated_seq, indel_outcomes = simulate_indel_event(
            mutated_seq,
            rng,
            probability=args.indel_probability,
            max_codon_length=args.indel_max_codons,
            proofreading_acceptance=args.indel_proofreading,
            codon_library=codon_library,
            feature_lookup=feature_lookup,
        )
        mutated_protein = mutated_seq.translate()
        silent = mutated_protein == original_protein

        classification = "silencieuse" if silent else "non-silencieuse"
        print(f"Simulation {sample_index}: mutation {classification}")

        affected_features: set[GenomeFeature] = set()
        accepted_features: list[tuple[MutationOutcome, GenomeFeature | None]] = []
        rejected_features: list[tuple[MutationOutcome, GenomeFeature | None]] = []
        for outcome in outcomes:
            feature = feature_lookup[outcome.position] if feature_lookup else None
            if outcome.accepted:
                accepted_features.append((outcome, feature))
                if feature:
                    affected_features.add(feature)
            else:
                rejected_features.append((outcome, feature))
                if feature:
                    feature_display_names.setdefault(feature.name, describe_feature(feature))
                    stats = feature_stats[feature.name]
                    stats["rejected"] += 1
                else:
                    feature_display_names.setdefault("non_annotée", describe_feature(None))
                    stats = feature_stats["non_annotée"]
                    stats["rejected"] += 1
        for indel in indel_outcomes:
            if indel.accepted and indel.feature:
                affected_features.add(indel.feature)

        for outcome, feature in accepted_features:
            effect = "=" if outcome.is_silent else "->"
            label = severity_label(
                outcome.severity_score,
                is_silent=outcome.is_silent,
                mutated_amino=outcome.mutated_amino_acid,
                original_amino=outcome.original_amino_acid,
            )
            feature_label = describe_feature(feature)
            flags_text = ",".join(outcome.flags) if outcome.flags else "aucun"
            print(
                f"  - [acceptée] nt {outcome.position}: {outcome.original_base}{effect}{outcome.mutated_base} | "
                f"codon {outcome.codon_index}: {outcome.original_codon}{effect}{outcome.mutated_codon} | "
                f"aa {outcome.original_amino_acid}{effect}{outcome.mutated_amino_acid} | "
                f"score {outcome.severity_score} ({label}) | "
                f"BLOSUM {outcome.blosum_score:.1f} | Grantham {outcome.grantham_distance:.0f} | "
                f"Δhydro {outcome.hydrophobicity_delta:+.2f} | p_accept {outcome.acceptance_probability:.2f} | "
                f"codon_pref {outcome.codon_preference:.2f} | flags {flags_text} | "
                f"protéine: {feature_label}"
            )

            feature_key = feature.name if feature else "non_annotée"
            feature_display_names.setdefault(feature_key, feature_label)
            stats = feature_stats[feature_key]
            stats["total"] += 1
            stats["severity_sum"] += outcome.severity_score
            stats["p_accept_sum"] += outcome.acceptance_probability
            stats["blosum_sum"] += outcome.blosum_score
            stats["grantham_sum"] += outcome.grantham_distance
            stats["hydro_sum"] += outcome.hydrophobicity_delta
            if outcome.is_silent:
                stats["syn"] += 1
                total_synonymous += 1
            else:
                stats["nonsyn"] += 1
                total_nonsynonymous += 1

            accepted_blosum_scores.append(outcome.blosum_score)
            accepted_probabilities.append(outcome.acceptance_probability)
            grantham_values.append(outcome.grantham_distance)
            hydrophobicity_deltas.append(outcome.hydrophobicity_delta)

            mutation_rows.append(
                " | ".join(
                    [
                        "[acceptée]",
                        f"nt {outcome.position}: {outcome.original_base}->{outcome.mutated_base}",
                        f"aa {outcome.original_amino_acid}->{outcome.mutated_amino_acid}",
                        f"score {outcome.severity_score} ({label})",
                        f"BLOSUM {outcome.blosum_score:.1f}",
                        f"Grantham {outcome.grantham_distance:.0f}",
                        f"Δhydro {outcome.hydrophobicity_delta:+.2f}",
                        f"p_accept {outcome.acceptance_probability:.2f}",
                        f"codon_pref {outcome.codon_preference:.2f}",
                        f"flags {flags_text}",
                        f"protéine {feature_label}",
                    ]
                )
            )

            if outcome.flags:
                stats["danger"] += 1
                joined_flags = ",".join(outcome.flags)
                red_flags.append(
                    f"nt {outcome.position}: {outcome.original_base}->{outcome.mutated_base} | "
                    f"aa {outcome.original_amino_acid}->{outcome.mutated_amino_acid} | "
                    f"flags {joined_flags} | "
                    f"protéine {feature_label}"
                )
            mutations_export_rows.append(
                {
                    "sample": sample_index,
                    "protein": feature_label,
                    "protein_id": feature.name if feature else "non_annotée",
                    "position": outcome.codon_index + 1,
                    "position_nt": outcome.position + 1,
                    "aa_ref": outcome.original_amino_acid,
                    "aa_mut": outcome.mutated_amino_acid,
                    "blosum": outcome.blosum_score,
                    "grantham": outcome.grantham_distance,
                    "p_accept": outcome.acceptance_probability,
                    "severity": outcome.severity_score,
                    "hydrophobicity_delta": outcome.hydrophobicity_delta,
                    "flags": ",".join(outcome.flags),
                }
            )
        if rejected_features:
            for outcome, feature in rejected_features:
                effect = "=" if outcome.is_silent else "->"
                label = severity_label(
                    outcome.severity_score,
                    is_silent=outcome.is_silent,
                    mutated_amino=outcome.mutated_amino_acid,
                    original_amino=outcome.original_amino_acid,
                )
                print(
                    f"  - [rejetée] nt {outcome.position}: {outcome.original_base}{effect}{outcome.mutated_base} | "
                    f"codon {outcome.codon_index}: {outcome.original_codon}{effect}{outcome.mutated_codon} | "
                    f"aa {outcome.original_amino_acid}{effect}{outcome.mutated_amino_acid} | "
                    f"score {outcome.severity_score} ({label}) | "
                    f"BLOSUM {outcome.blosum_score:.1f} | Grantham {outcome.grantham_distance:.0f} | "
                    f"Δhydro {outcome.hydrophobicity_delta:+.2f} | p_accept {outcome.acceptance_probability:.2f} | "
                    f"codon_pref {outcome.codon_preference:.2f} | flags {','.join(outcome.flags) if outcome.flags else 'aucun'} | "
                    f"protéine: {describe_feature(feature)}"
                )

                feature_key = feature.name if feature else "non_annotée"
                feature_display_names.setdefault(feature_key, describe_feature(feature))
                rejected_blosum_scores.append(outcome.blosum_score)
                rejected_probabilities.append(outcome.acceptance_probability)
                total_rejected += 1
                rejected_rows.append(
                    " | ".join(
                        [
                            "[rejetée]",
                            f"nt {outcome.position}: {outcome.original_base}->{outcome.mutated_base}",
                            f"aa {outcome.original_amino_acid}->{outcome.mutated_amino_acid}",
                            f"score {outcome.severity_score} ({label})",
                            f"BLOSUM {outcome.blosum_score:.1f}",
                            f"Grantham {outcome.grantham_distance:.0f}",
                            f"Δhydro {outcome.hydrophobicity_delta:+.2f}",
                            f"p_accept {outcome.acceptance_probability:.2f}",
                            f"codon_pref {outcome.codon_preference:.2f}",
                            f"flags {','.join(outcome.flags) if outcome.flags else 'aucun'}",
                            f"protéine {describe_feature(feature)}",
                        ]
                    )
                )
        if indel_outcomes:
            for indel in indel_outcomes:
                status = "acceptée" if indel.accepted else "rejetée"
                feature = describe_feature(indel.feature)
                length_codons = indel.length_nt // 3
                print(
                    f"  - indel {indel.kind} ({status}): position {indel.position}, "
                    f"{length_codons} codon(s) | protéine: {feature}"
                )
                indel_records.append(
                    f"indel {indel.kind} ({status}) | nt {indel.position} | {length_codons} codon(s) | protéine {feature}"
                )
                if indel.accepted and indel.feature:
                    feature_key = indel.feature.name
                    feature_display_names.setdefault(feature_key, feature)
                    stats = feature_stats[feature_key]
                    stats["indels"] += 1
        if affected_features:
            summary = ", ".join(sorted(describe_feature(f) for f in affected_features))
            print(f"  Protéine(s) ciblée(s): {summary}")
        else:
            print("  Protéine(s) ciblée(s): non annotée")
        if args.preview_aa:
            preview = str(mutated_protein)[: args.preview_aa]
            suffix = "..." if len(mutated_protein) > args.preview_aa else ""
            print(f"  Aperçu protéique: {preview}{suffix}")
        print(f"  Protéine identique: {'oui' if silent else 'non'}")

        synonymous = sum(1 for outcome in outcomes if outcome.accepted and outcome.is_silent)
        nonsynonymous = sum(1 for outcome in outcomes if outcome.accepted and not outcome.is_silent)
        if synonymous:
            ratio = nonsynonymous / synonymous
            print(f"  Ratio dN/dS: {ratio:.2f}")
        elif nonsynonymous:
            print("  Ratio dN/dS: inf (aucune substitution synonyme)")
        else:
            print("  Ratio dN/dS: NA (aucune substitution enregistrée)")
        print("-")

        if args.protein_fasta:
            accepted_labels = [
                f"nt{outcome.position}{outcome.original_base}>{outcome.mutated_base}"
                for outcome in outcomes
                if outcome.accepted
            ]
            description = ", ".join(accepted_labels) or "aucune mutation"
            for indel in indel_outcomes:
                if indel.accepted:
                    description += f", indel-{indel.kind}{indel.position}"
            if affected_features:
                fasta_id = "+".join(sorted(sanitize_identifier(feature.name) for feature in affected_features))
            else:
                fasta_id = f"mutation_{sample_index}"
            fasta_id = f"{sequence_record.id}_{fasta_id}" if sequence_record.id else fasta_id
            protein_records.append(
                SeqRecord(
                    mutated_protein,
                    id=fasta_id,
                    description=description,
                )
            )

    print("=" * 40)
    print("Résumé global")
    if total_synonymous or total_nonsynonymous:
        if total_synonymous:
            global_ratio = total_nonsynonymous / total_synonymous
            print(f"  dN/dS global: {global_ratio:.2f}")
        elif total_nonsynonymous:
            print("  dN/dS global: inf (aucune substitution synonyme)")
    if accepted_blosum_scores:
        mean_blosum = sum(accepted_blosum_scores) / len(accepted_blosum_scores)
        print(
            f"  BLOSUM accepté: moyenne {mean_blosum:.2f}, min {min(accepted_blosum_scores):.1f}, max {max(accepted_blosum_scores):.1f}"
        )
    if rejected_blosum_scores:
        mean_blosum_rejected = sum(rejected_blosum_scores) / len(rejected_blosum_scores)
        print(
            f"  BLOSUM rejeté: moyenne {mean_blosum_rejected:.2f}, min {min(rejected_blosum_scores):.1f}, max {max(rejected_blosum_scores):.1f}"
        )
    if accepted_probabilities:
        mean_accept = sum(accepted_probabilities) / len(accepted_probabilities)
        print(
            f"  p_accept accepté: moyenne {mean_accept:.2f}, min {min(accepted_probabilities):.2f}, max {max(accepted_probabilities):.2f}"
        )
    if rejected_probabilities:
        mean_reject = sum(rejected_probabilities) / len(rejected_probabilities)
        print(
            f"  p_accept rejeté: moyenne {mean_reject:.2f}, min {min(rejected_probabilities):.2f}, max {max(rejected_probabilities):.2f}"
        )
    if total_rejected:
        print(f"  Mutations rejetées: {total_rejected}")
    if grantham_values:
        mean_grantham = sum(grantham_values) / len(grantham_values)
        print(
            f"  Grantham: moyenne {mean_grantham:.1f}, min {min(grantham_values):.0f}, max {max(grantham_values):.0f}"
        )
    if hydrophobicity_deltas:
        mean_hydro = sum(hydrophobicity_deltas) / len(hydrophobicity_deltas)
        print(
            f"  Δhydrophobicité moyenne: {mean_hydro:+.2f} (min {min(hydrophobicity_deltas):+.2f}, max {max(hydrophobicity_deltas):+.2f})"
        )
    if mutation_rows:
        print("  Mutations acceptées (mutation | type | sévérité | protéine):")
        for row in mutation_rows:
            print(f"    {row}")
    if rejected_rows:
        print("  Mutations rejetées (propositions filtrées):")
        for row in rejected_rows:
            print(f"    {row}")
    if feature_stats:
        print("  Mutations par protéine:")
        for feature_key in sorted(feature_stats.keys()):
            stats = feature_stats[feature_key]
            total = stats["total"]
            rejected = stats["rejected"]
            if total == 0 and stats["indels"] == 0 and rejected == 0:
                continue
            display_name = feature_display_names.get(feature_key, feature_key)
            severity_mean = stats["severity_sum"] / total if total else 0.0
            p_accept_mean = stats["p_accept_sum"] / total if total else 0.0
            blosum_mean = stats["blosum_sum"] / total if total else 0.0
            grantham_mean = stats["grantham_sum"] / total if total else 0.0
            hydro_mean = stats["hydro_sum"] / total if total else 0.0
            length_codons = feature_lengths.get(feature_key)
            if length_codons:
                tolerance = stats["total"] / length_codons
                tolerance_text = f"tolérance {tolerance:.3f} mut./aa (longueur {length_codons} aa)"
            else:
                tolerance_text = "tolérance NA (longueur non déterminée)"
            print(
                f"    - {display_name}: {total} mutations acceptées ({stats['nonsyn']} non synonymes, {stats['syn']} synonymes), "
                f"{rejected} rejetées, {tolerance_text}, sévérité moyenne {severity_mean:.2f}, p_accept moyenne {p_accept_mean:.2f}, "
                f"BLOSUM moyen {blosum_mean:.2f}, Grantham moyen {grantham_mean:.1f}, Δhydro moyen {hydro_mean:+.2f}, "
                f"flags critiques {stats['danger']}, indels acceptés {stats['indels']}"
            )
    if red_flags:
        print("  Liste rouge (stop / Pro / Cys / Gly impactés):")
        for item in red_flags:
            print(f"    {item}")
    if indel_records:
        print("  Indels simulés:")
        for record in indel_records:
            print(f"    {record}")

    if args.mutations_csv:
        args.mutations_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "sample",
            "protein",
            "protein_id",
            "position",
            "position_nt",
            "aa_ref",
            "aa_mut",
            "blosum",
            "grantham",
            "p_accept",
            "severity",
            "hydrophobicity_delta",
            "flags",
        ]
        with args.mutations_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mutations_export_rows)
        print(f"Mutations acceptées enregistrées dans {args.mutations_csv}")

    if args.protein_fasta:
        SeqIO.write(protein_records, args.protein_fasta, "fasta")
        print(f"Séquences protéiques enregistrées dans {args.protein_fasta}")


if __name__ == "__main__":
    main()
