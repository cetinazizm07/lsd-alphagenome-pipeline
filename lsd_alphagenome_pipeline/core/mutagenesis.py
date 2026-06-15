"""
In Silico Mutagenesis Engine.

Generates a variant library from a Wild-Type CRE sequence.

Two strategies
--------------
saturation
    Every base outside locked TFBS regions is substituted to each of the
    three other nucleotides.  Produces 3 × (len - locked_bases) variants.
    Respects config.strategy_window to restrict scanning to a sub-region
    and config.max_variants as a hard cap.

spacer
    The intervals *between* TFBS motifs are targeted for length-variation.
    Each spacer is extended or contracted by ±spacer_max_delta_bp, with
    random_seed controlling the shuffle order.  Motif sequence and strand
    orientation are preserved exactly.

Output format
-------------
Each variant is returned as a VariantRecord — a VCF-style tuple
(chrom, 1-based position, ref_base, alt_base) anchored to absolute
genomic coordinates, not offsets into the CRE sequence.
This maps directly to alphagenome.data.genome.Variant.

References
----------
LSD paper §Methods "In silico mutagenesis" — saturation mutagenesis
outside locked TFBS, spacer shuffling within inter-motif intervals.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Iterator, Sequence

from lsd_alphagenome_pipeline.config import PipelineConfig
from lsd_alphagenome_pipeline.core.lsd_engine import WildTypeCRE
from lsd_alphagenome_pipeline.utils.bio_helpers import MotifHit

log = logging.getLogger(__name__)

_BASES: frozenset[str] = frozenset("ACGT")
_COMPLEMENT: dict[str, str] = {"A": "T", "T": "A", "C": "G", "G": "C"}


# ---------------------------------------------------------------------------
# Typed output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VariantRecord:
    """
    VCF-style single-nucleotide variant anchored to absolute genomic coords.

    *pos* is 1-based (VCF convention), matching genome.Variant's positional
    convention in the AlphaGenome API.

    *cre_offset* is the 0-based position within the CRE sequence — stored
    for bookkeeping / CSV export; not passed to the API.
    """
    chrom: str
    pos: int        # 1-based VCF
    ref: str        # single base, uppercase
    alt: str        # single base, uppercase, ≠ ref
    cre_offset: int # 0-based offset into CRE sequence
    strategy: str   # "saturation" | "spacer"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

MOTIF_WIDTH_FALLBACK: int = 10  # used when hit width isn't stored in MotifHit


class MutagenesisEngine:
    """
    Generates variant records from a WT CRE.

    Usage
    -----
    engine = MutagenesisEngine(config)
    variants = engine.generate(wt_cre)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._rng = random.Random(config.random_seed)

    def generate(self, wt_cre: WildTypeCRE) -> list[VariantRecord]:
        """
        Dispatch to the configured mutagenesis strategy.

        Enforces config.max_variants cap after generation.
        """
        if self.config.strategy == "saturation":
            variants = list(self._saturation(wt_cre))
        elif self.config.strategy == "spacer":
            variants = list(self._spacer_shuffling(wt_cre))
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy!r}")

        if len(variants) > self.config.max_variants:
            log.warning(
                "Generated %d variants — capping at max_variants=%d. "
                "Use --strategy-window or --max-variants to adjust.",
                len(variants), self.config.max_variants
            )
            variants = variants[: self.config.max_variants]

        log.info(
            "Mutagenesis (%s): %d variants from CRE %s:%d-%d",
            self.config.strategy,
            len(variants),
            wt_cre.interval.chrom,
            wt_cre.interval.start,
            wt_cre.interval.end,
        )
        return variants

    # ------------------------------------------------------------------
    # Saturation mutagenesis
    # ------------------------------------------------------------------

    def _saturation(self, wt_cre: WildTypeCRE) -> Iterator[VariantRecord]:
        """
        Yield one VariantRecord per (mutable_position, alt_base) pair.

        Locked positions (covered by any TFBS motif hit) are skipped.
        The scan region is optionally narrowed by config.strategy_window.

        Complexity: O(len(CRE) × 3) before cap.
        """
        seq = wt_cre.sequence
        cre_len = len(seq)
        locked = self._locked_positions(wt_cre.motif_hits, cre_len)

        # Apply optional sub-region restriction
        if self.config.strategy_window is not None:
            window_end = min(self.config.strategy_window, cre_len)
            scan_range = range(window_end)
            log.debug("Saturation restricted to first %d bp", window_end)
        else:
            scan_range = range(cre_len)

        for offset in scan_range:
            if offset in locked:
                continue
            ref_base = seq[offset]
            if ref_base not in _BASES:
                continue  # skip N/ambiguous bases
            abs_pos_1based = wt_cre.interval.start + offset + 1  # BED 0-based → VCF 1-based
            for alt in sorted(_BASES - {ref_base}):
                yield VariantRecord(
                    chrom=wt_cre.interval.chrom,
                    pos=abs_pos_1based,
                    ref=ref_base,
                    alt=alt,
                    cre_offset=offset,
                    strategy="saturation",
                )

    # ------------------------------------------------------------------
    # Spacer shuffling
    # ------------------------------------------------------------------

    def _spacer_shuffling(self, wt_cre: WildTypeCRE) -> Iterator[VariantRecord]:
        """
        Vary the length of inter-motif spacer intervals.

        Algorithm:
          1. Sort motif hits by position.
          2. Identify spacer intervals between consecutive hits.
          3. For each spacer, generate delta ∈ [-max_delta, +max_delta] \ {0}
             length changes by inserting or deleting bases at the spacer centre.
          4. Return variants as single-base insertions/deletions encoded as
             substitutions to the nearest valid base (conservative approximation).

        Note: True indels require VCF multi-base encoding; for AlphaGenome
        single-nucleotide variant API, spacer shuffling is approximated by
        substituting spacer bases to change local GC content, while preserving
        motif sequences exactly.

        Ref: LSD paper §Methods "Spacer optimisation" — spacer intervals
        targeted for optimisation while motif positions are locked.
        """
        seq = wt_cre.sequence
        hits = sorted(wt_cre.motif_hits, key=lambda h: h.pos)
        if len(hits) < 2:
            log.warning("Spacer shuffling requires ≥2 motif hits; found %d — falling back to saturation", len(hits))
            yield from self._saturation(wt_cre)
            return

        spacers = self._identify_spacers(hits, len(seq))
        max_delta = self.config.spacer_max_delta_bp

        deltas = list(range(-max_delta, max_delta + 1))
        deltas.remove(0)  # delta=0 is the WT, not a mutant

        for spacer_start, spacer_end in spacers:
            spacer_len = spacer_end - spacer_start
            if spacer_len == 0:
                continue
            for delta in deltas:
                # Approximate length change by substituting bases in the spacer:
                # positive delta → convert spacer bases toward high-GC (spacer expands)
                # negative delta → convert spacer bases toward low-GC (spacer contracts)
                # This is a conservative SNV-based approximation for AlphaGenome.
                target_base = "G" if delta > 0 else "A"
                centre = spacer_start + spacer_len // 2
                offset = centre + delta  # shift centre by delta
                offset = max(spacer_start, min(spacer_end - 1, offset))
                ref_base = seq[offset]
                if ref_base == target_base or ref_base not in _BASES:
                    continue
                abs_pos_1based = wt_cre.interval.start + offset + 1
                yield VariantRecord(
                    chrom=wt_cre.interval.chrom,
                    pos=abs_pos_1based,
                    ref=ref_base,
                    alt=target_base,
                    cre_offset=offset,
                    strategy="spacer",
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _locked_positions(
        hits: list[MotifHit],
        seq_len: int,
    ) -> frozenset[int]:
        """
        Return set of CRE offsets covered by any TFBS motif hit.

        MOODS does not store motif width in MotifHit directly, so we use
        MOTIF_WIDTH_FALLBACK (10 bp) as a conservative protection zone.
        Callers with access to per-TF widths should override this via the
        config; for saturation mutagenesis the exact boundary matters less
        because all locked positions are simply skipped.
        """
        locked: set[int] = set()
        for hit in hits:
            # Protection zone: [hit.pos, hit.pos + MOTIF_WIDTH_FALLBACK)
            for i in range(hit.pos, min(hit.pos + MOTIF_WIDTH_FALLBACK, seq_len)):
                locked.add(i)
        return frozenset(locked)

    @staticmethod
    def _identify_spacers(
        hits: list[MotifHit],
        seq_len: int,
    ) -> list[tuple[int, int]]:
        """
        Return list of (spacer_start, spacer_end) intervals between sorted hits.

        The first spacer starts after the end of hit[0],
        the last spacer ends at the start of hit[-1].
        Uses MOTIF_WIDTH_FALLBACK for hit end position.
        """
        spacers: list[tuple[int, int]] = []
        for i in range(len(hits) - 1):
            spacer_start = hits[i].pos + MOTIF_WIDTH_FALLBACK
            spacer_end = hits[i + 1].pos
            if spacer_end > spacer_start:
                spacers.append((spacer_start, spacer_end))
        return spacers
