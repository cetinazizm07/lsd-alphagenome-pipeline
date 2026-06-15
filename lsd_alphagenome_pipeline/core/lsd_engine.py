"""
LSD CRE selection engine.

Re-implements the three-step LSD framework from:
  Gargiulo et al., Nature Communications 2024
  "Logical design of synthetic cis-regulatory DNA"
  https://gitlab.com/gargiulo_lab/LSD_framework

Step 1 — Locus Definition
    For each signature gene in the input BED, define a genomic locus bounded
    by CTCF insulator sites (or ±locus_flank_bp if no CTCF BED is supplied).

Step 2 — TFBS Assessment
    Scan each locus for transcription factor binding sites (TFBS) using the
    MOODS motif scanner (replacing the original FIMO binary).
    Motifs are provided in MEME format (JASPAR / HOCOMOCO).

Step 3 — CRE Selection via Combinatorial Enrichment Matrix
    Score candidate CRE windows by the number of distinct TFs represented
    and the density of motif hits.  The window with the highest composite
    score is returned as the Wild-Type (WT) CRE.

    Composite score (per window W):
        S(W) = Σ_tf [ I(tf ∈ W) × density_tf(W) ]
    where density_tf(W) = count_tf_hits_in_W / len(W_bp)

    This mirrors the enrichment matrix described in the LSD paper §Methods
    "CRE selection" — each row is a TF, each column is a candidate window;
    the column with maximum row-sum is selected.

Reproducibility notes
----------------------
* MOODS p-value threshold defaults to 1e-4 (LSD paper §Methods)
* Window size is configurable (default 500 bp)
* All coordinates are 0-based half-open throughout
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from lsd_alphagenome_pipeline.config import PipelineConfig
from lsd_alphagenome_pipeline.utils.bio_helpers import (
    GenomicInterval,
    MotifHit,
    extract_sequence,
    load_fasta_index,
    motif_hit_to_absolute,
    parse_bed,
    scan_motifs,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class LocusRecord:
    """A signature gene locus with its CTCF-bounded extent."""
    gene: GenomicInterval          # original gene entry from BED
    locus: GenomicInterval         # CTCF-bounded or fallback-padded region


@dataclass
class WildTypeCRE:
    """
    The optimal natural CRE selected by LSD.

    *interval* is absolute genomic coordinates (0-based, half-open).
    *sequence* is the extracted DNA string from the reference FASTA.
    *motif_hits* are the TFBS hits within the CRE, with positions relative
    to the start of *interval*.
    *enrichment_score* is the composite TF-density score.
    """
    interval: GenomicInterval
    sequence: str
    motif_hits: list[MotifHit]
    enrichment_score: float
    locus: LocusRecord


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class LSDEngine:
    """
    Implements the three-step LSD CRE selection framework.

    Usage
    -----
    engine = LSDEngine(config)
    wt_cre = engine.select_wt_cre()
    """

    # Sliding window parameters for Step 3.
    # Ref: LSD framework src/sLCR_functions.py sliding_window() —
    # "Creates 150bp windows sliding by 50bp increments"
    WINDOW_BP: int = 150
    STEP_BP: int = 50

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._fasta_index: dict = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_wt_cre(self) -> WildTypeCRE:
        """
        Run all three LSD steps and return the highest-scoring CRE.

        Raises RuntimeError if no valid CRE window can be scored
        (e.g., locus is shorter than WINDOW_BP or no motifs found).
        """
        log.info("LSD Step 1: Locus definition")
        loci = self._step1_define_loci()

        log.info("LSD Step 2: TFBS assessment across %d loci", len(loci))
        scored_windows = self._step2_scan_and_score(loci)

        if not scored_windows:
            raise RuntimeError(
                "No scoreable CRE windows found.  "
                "Check that your motif file contains valid MEME matrices "
                "and that loci are at least %d bp wide." % self.WINDOW_BP
            )

        log.info("LSD Step 3: CRE selection from %d candidate windows", len(scored_windows))
        best = max(scored_windows, key=lambda x: x[2])  # (locus, window_interval, score, hits)
        locus_rec, window, score, hits = best

        log.info(
            "Selected WT CRE: %s:%d-%d  score=%.4f  motif_hits=%d",
            window.chrom, window.start, window.end, score, len(hits)
        )

        sequence = self._extract(window)
        return WildTypeCRE(
            interval=window,
            sequence=sequence,
            motif_hits=hits,
            enrichment_score=score,
            locus=locus_rec,
        )

    # ------------------------------------------------------------------
    # Step 1 — Locus Definition
    # ------------------------------------------------------------------

    def _step1_define_loci(self) -> list[LocusRecord]:
        """
        For each gene in the signature BED, find its CTCF-bounded locus.

        If a CTCF BED is supplied (config.ctcf_bed), the locus is bounded by
        the nearest upstream and downstream CTCF sites.  Otherwise, a fixed
        flank of config.locus_flank_bp is applied on each side of the gene.

        Ref: LSD paper §Methods "Locus definition" — CTCF sites are used as
        topological boundaries because they mark CTCF-cohesin loop anchors.
        """
        genes = parse_bed(self.config.genes_bed, min_cols=3)
        if not genes:
            raise ValueError(f"No genes found in {self.config.genes_bed}")

        ctcf_sites: list[GenomicInterval] = []
        if self.config.ctcf_bed is not None:
            ctcf_sites = parse_bed(self.config.ctcf_bed, min_cols=3)
            log.info("Loaded %d CTCF sites", len(ctcf_sites))

        records: list[LocusRecord] = []
        for gene in genes:
            locus = self._bound_by_ctcf(gene, ctcf_sites)
            records.append(LocusRecord(gene=gene, locus=locus))
            log.debug(
                "Gene %s → locus %s:%d-%d (%d bp)",
                gene.name, locus.chrom, locus.start, locus.end, locus.length
            )

        return records

    def _bound_by_ctcf(
        self,
        gene: GenomicInterval,
        ctcf_sites: list[GenomicInterval],
    ) -> GenomicInterval:
        """
        Return a locus interval bounded by the nearest flanking CTCF sites.

        If no CTCF sites on the same chromosome exist within locus_flank_bp,
        the fixed flank is used as fallback.

        Algorithm:
          upstream_ctcf  = rightmost CTCF site to the left of gene.start
          downstream_ctcf = leftmost CTCF site to the right of gene.end
        """
        flank = self.config.locus_flank_bp
        default_start = max(0, gene.start - flank)
        default_end = gene.end + flank

        same_chrom = [s for s in ctcf_sites if s.chrom == gene.chrom]
        if not same_chrom:
            return GenomicInterval(gene.chrom, default_start, default_end, gene.name)

        upstream = [s for s in same_chrom if s.end <= gene.start and s.end >= default_start]
        downstream = [s for s in same_chrom if s.start >= gene.end and s.start <= default_end]

        locus_start = max(s.start for s in upstream) if upstream else default_start
        locus_end = min(s.end for s in downstream) if downstream else default_end

        return GenomicInterval(gene.chrom, locus_start, locus_end, gene.name)

    # ------------------------------------------------------------------
    # Step 2 — TFBS Assessment + Step 3 — Scoring
    # ------------------------------------------------------------------

    def _step2_scan_and_score(
        self,
        loci: list[LocusRecord],
    ) -> list[tuple[LocusRecord, GenomicInterval, float, list[MotifHit]]]:
        """
        Scan each locus with sliding windows; return (locus, window, score, hits)
        for every window that contains at least one motif hit.

        Window scan parameters:
          width = WINDOW_BP (500 bp)
          step  = STEP_BP   (50 bp)

        These match the resolution used in the LSD paper enrichment matrix.
        """
        fasta = self._load_fasta()
        results: list[tuple[LocusRecord, GenomicInterval, float, list[MotifHit]]] = []

        for locus_rec in loci:
            locus = locus_rec.locus
            if locus.length < self.WINDOW_BP:
                log.warning(
                    "Locus %s is only %d bp — shorter than window (%d bp), skipping",
                    locus.name, locus.length, self.WINDOW_BP
                )
                continue

            # Extract full locus sequence once; scan windows as sub-strings.
            locus_seq = extract_sequence(fasta, locus)
            locus_hits = scan_motifs(
                sequence=locus_seq,
                motif_meme_path=self.config.tf_motifs,
                pvalue_threshold=self.config.motif_pvalue_threshold,
                bg_order=self.config.motif_bg_order,
            )
            log.debug(
                "Locus %s: %d hits across %d bp",
                locus.name, len(locus_hits), locus.length
            )

            # Slide window across locus
            for win_start in range(0, locus.length - self.WINDOW_BP + 1, self.STEP_BP):
                win_end = win_start + self.WINDOW_BP
                window_hits = [
                    h for h in locus_hits
                    if win_start <= h.pos < win_end
                ]
                if not window_hits:
                    continue

                score = self._enrichment_score(window_hits, self.WINDOW_BP)
                abs_interval = GenomicInterval(
                    chrom=locus.chrom,
                    start=locus.start + win_start,
                    end=locus.start + win_end,
                    name=f"{locus.name}_w{win_start}",
                )
                results.append((locus_rec, abs_interval, score, window_hits))

        return results

    @staticmethod
    def _enrichment_score(hits: list[MotifHit], window_bp: int) -> float:
        """
        Composite enrichment score for a candidate window.

        S(W) = Σ_tf [ I(tf ∈ W) × hits_per_tf / window_bp ]

        Each TF contributes its normalised hit density.  Using density rather
        than raw count penalises windows dominated by a single repetitive TF.

        Ref: LSD paper §Methods "CRE selection" — enrichment matrix rows
        correspond to TFs, columns to windows; the column with max row-sum wins.
        """
        from collections import Counter
        counts: Counter[str] = Counter(h.tf_name for h in hits)
        return sum(c / window_bp for c in counts.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_fasta(self) -> dict:
        if not self._fasta_index:
            if not self.config.fasta.exists():
                raise FileNotFoundError(
                    f"Reference FASTA not found: {self.config.fasta}\n"
                    "Download from UCSC or Ensembl and set --fasta, or place at "
                    "reference/genome.fa relative to your working directory."
                )
            self._fasta_index = load_fasta_index(self.config.fasta)
        return self._fasta_index

    def _extract(self, interval: GenomicInterval) -> str:
        return extract_sequence(self._load_fasta(), interval)
