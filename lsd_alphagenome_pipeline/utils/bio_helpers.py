"""
Genomic I/O utilities.

* FASTA extraction  — Biopython SeqIO + pysam-style index
* BED parsing       — returns typed dataclasses, avoids raw string dicts
* MOODS scanning    — pure-Python TFBS motif scanning (no FIMO binary)

References
----------
* Biopython SeqIO: https://biopython.org/wiki/SeqIO
* MOODS-python:   https://github.com/jhkorhonen/MOODS/tree/master/python
* LSD paper §Methods "TFBS assessment" — p-value ≤ 1e-4, 1st-order Markov bg
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import MOODS.scan      # type: ignore[import]
import MOODS.tools     # type: ignore[import]
import MOODS.parsers   # type: ignore[import]
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed genomic primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenomicInterval:
    """Half-open [start, end) on a chromosome, 0-based (BED convention)."""
    chrom: str
    start: int
    end: int
    name: str = ""
    strand: str = "."

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def midpoint(self) -> int:
        return (self.start + self.end) // 2

    def expand(self, bp: int) -> "GenomicInterval":
        """Return interval padded by *bp* on each side."""
        return GenomicInterval(
            chrom=self.chrom,
            start=max(0, self.start - bp),
            end=self.end + bp,
            name=self.name,
            strand=self.strand,
        )


@dataclass(frozen=True)
class MotifHit:
    """
    Single TFBS match from MOODS.

    *pos* is 0-based offset from the start of the scanned sequence.
    *strand* is '+' or '-'.
    """
    tf_name: str
    pos: int        # 0-based, relative to scanned sequence
    strand: str
    score: float
    pvalue: float


# ---------------------------------------------------------------------------
# FASTA extraction
# ---------------------------------------------------------------------------

def load_fasta_index(fasta_path: Path) -> dict[str, SeqRecord]:
    """
    Load all chromosomes from *fasta_path* into memory.

    For large genomes prefer pysam.FastaFile; this implementation targets
    smaller / filtered reference FASTAs used in testing.

    Note: caller is responsible for verifying assembly — chromosome names
    in the FASTA must match those in the BED files.
    """
    records: dict[str, SeqRecord] = {}
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        records[rec.id] = rec
    log.debug("Loaded %d chromosomes from %s", len(records), fasta_path)
    return records


def extract_sequence(
    fasta_index: dict[str, SeqRecord],
    interval: GenomicInterval,
) -> str:
    """
    Return uppercase DNA string for *interval* from *fasta_index*.

    Raises KeyError if the chromosome is absent — signals assembly mismatch
    rather than silently returning empty sequence.
    """
    if interval.chrom not in fasta_index:
        raise KeyError(
            f"Chromosome '{interval.chrom}' not found in FASTA. "
            "Verify --assembly matches your BED file coordinates."
        )
    rec = fasta_index[interval.chrom]
    seq = str(rec.seq[interval.start:interval.end]).upper()
    if len(seq) != interval.length:
        raise ValueError(
            f"FASTA returned {len(seq)} bp for {interval} "
            f"(expected {interval.length} bp — possible truncation at chromosome end)"
        )
    return seq


# ---------------------------------------------------------------------------
# BED parsing
# ---------------------------------------------------------------------------

def parse_bed(bed_path: Path, min_cols: int = 3) -> list[GenomicInterval]:
    """
    Parse a BED file into GenomicInterval list.

    Supports 3-col (chrom, start, end) and 6-col (+ name, score, strand).
    Comment lines (#) and empty lines are skipped.

    Ref: UCSC BED format — https://genome.ucsc.edu/FAQ/FAQformat.html#format1
    """
    intervals: list[GenomicInterval] = []
    with open(bed_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            parts = line.split("\t")
            if len(parts) < min_cols:
                raise ValueError(
                    f"{bed_path}:{lineno} — expected ≥{min_cols} tab-separated "
                    f"columns, got {len(parts)}"
                )
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            name = parts[3] if len(parts) > 3 else ""
            strand = parts[5] if len(parts) > 5 else "."
            intervals.append(GenomicInterval(chrom, start, end, name, strand))
    log.debug("Parsed %d intervals from %s", len(intervals), bed_path)
    return intervals


# ---------------------------------------------------------------------------
# MOODS motif scanning
# ---------------------------------------------------------------------------

def scan_motifs(
    sequence: str,
    motif_meme_path: Path,
    pvalue_threshold: float = 1e-4,
    bg_order: int = 1,
) -> list[MotifHit]:
    """
    Scan *sequence* for all motifs in *motif_meme_path* (MEME format).

    Uses MOODS in place of the FIMO binary for portability.  Parameter
    defaults replicate the LSD paper's TFBS assessment settings:
      - p-value ≤ 1e-4  (LSD paper §Methods "TFBS assessment")
      - 1st-order Markov background model

    Returns hits sorted by genomic position (ascending).

    MOODS API reference:
      MOODS.parsers.pfm_to_log_odds()  — convert PFM → log-odds matrix
      MOODS.tools.threshold_from_p()   — convert p-value → score threshold
      MOODS.scan.scan_dna()            — perform the scan
    """
    matrices, names = _load_meme_as_moods(motif_meme_path)
    if not matrices:
        raise ValueError(f"No motif matrices parsed from {motif_meme_path}")

    # Background nucleotide frequencies from the sequence itself (Markov order bg_order)
    bg = MOODS.tools.bg_from_sequence_dna(sequence, bg_order)

    log_odds = [MOODS.parsers.pfm_to_log_odds(m, bg) for m in matrices]
    thresholds = [
        MOODS.tools.threshold_from_p(lo, bg, pvalue_threshold)
        for lo in log_odds
    ]

    # scan_dna returns list[list[MOODS.scan.match]] — one inner list per matrix
    raw_results = MOODS.scan.scan_dna(sequence, log_odds, bg, thresholds)

    hits: list[MotifHit] = []
    for matrix_idx, (name, matches) in enumerate(zip(names, raw_results)):
        width = len(matrices[matrix_idx][0])  # number of columns in PFM
        for match in matches:
            pval = MOODS.tools.p_from_score(log_odds[matrix_idx], bg, match.score)
            strand = "+" if match.strand else "-"
            hits.append(MotifHit(
                tf_name=name,
                pos=match.pos,
                strand=strand,
                score=match.score,
                pvalue=pval,
            ))

    hits.sort(key=lambda h: h.pos)
    log.debug("Found %d motif hits in sequence of length %d", len(hits), len(sequence))
    return hits


def motif_hit_to_absolute(hit: MotifHit, locus: GenomicInterval) -> tuple[int, int]:
    """
    Convert a relative *MotifHit.pos* to absolute genomic coordinates.

    Returns (abs_start, abs_end) — half-open, 0-based, matching BED convention.
    """
    abs_start = locus.start + hit.pos
    # MOODS stores width implicitly; caller must supply if needed.
    # This function returns only the start anchor; width is TF-specific.
    return abs_start, abs_start


# ---------------------------------------------------------------------------
# Internal MEME → MOODS matrix parser
# ---------------------------------------------------------------------------

def _load_meme_as_moods(
    meme_path: Path,
) -> tuple[list[list[list[float]]], list[str]]:
    """
    Parse a MEME-format motif file and return (pfm_matrices, names).

    MEME format spec: https://meme-suite.org/meme/doc/meme-format.html
    Each PFM is shaped [4][width] — rows are A, C, G, T.

    MOODS expects matrices in column-major order: list of 4 lists (one per
    nucleotide), each of length W (motif width).
    """
    matrices: list[list[list[float]]] = []
    names: list[str] = []
    current_name: str | None = None
    current_rows: list[list[float]] = []
    in_matrix = False

    with open(meme_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("MOTIF"):
                if current_name and current_rows:
                    matrices.append(_rows_to_cols(current_rows))
                    names.append(current_name)
                current_name = line.split()[1]
                current_rows = []
                in_matrix = False
            elif "letter-probability matrix" in line:
                in_matrix = True
            elif in_matrix and line and not line.startswith("URL") and not line.startswith("Background"):
                try:
                    vals = [float(x) for x in line.split()]
                    if len(vals) == 4:
                        current_rows.append(vals)
                except ValueError:
                    in_matrix = False

    if current_name and current_rows:
        matrices.append(_rows_to_cols(current_rows))
        names.append(current_name)

    return matrices, names


def _rows_to_cols(rows: list[list[float]]) -> list[list[float]]:
    """Transpose row-major PFM [[A,C,G,T], ...] to column-major [[A_col], [C_col], [G_col], [T_col]]."""
    if not rows:
        return [[], [], [], []]
    return [[row[i] for row in rows] for i in range(4)]
