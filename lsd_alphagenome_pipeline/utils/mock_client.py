"""
Mock AlphaGenome client for demonstration without an API key.

Generates biologically-motivated synthetic ATAC-seq and RNA-seq scores
by simulating the effect of mutations on TF binding affinity.

Scientific rationale
--------------------
A mutation at a TFBS position disrupts the TF's consensus sequence,
reducing binding affinity → lower chromatin accessibility (ATAC) and
transcriptional output (RNA).  Mutations outside TFBS have smaller effects.

This mock replicates the expected directionality of effects observed in
the Gargiulo et al. 2024 paper saturation mutagenesis experiments.

Usage
-----
Replace AlphaGenomeClient with MockAlphaGenomeClient in AlphaGenomeValidator
by passing --mock to the CLI.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np


class _MockOutputs:
    """Mimics the attribute interface of AlphaGenome predict_variant outputs."""

    def __init__(self, atac_track: np.ndarray, rna_track: np.ndarray) -> None:
        self.atac_seq = atac_track
        self.rna_seq = rna_track


class MockAlphaGenomeClient:
    """
    Drop-in replacement for AlphaGenomeClient that requires no API key.

    Fitness model
    -------------
    WT baseline: ATAC=0.80, RNA=0.75 (typical accessible + expressed locus).

    Mutation effects (applied multiplicatively to the CRE-region mean):
      - Hit at a TFBS position: Δ = −0.30 ± 0.05 (disrupts TF binding)
      - Hit in spacer:          Δ = −0.05 ± 0.10 (minor structural effect)
      - Synonymous or neutral:  Δ =  0.00 ± 0.02 (noise floor)
    """

    WT_ATAC: float = 0.80
    WT_RNA: float = 0.75
    TRACK_LENGTH: int = 1_048_576  # AlphaGenome window size

    def __init__(self, locked_positions: frozenset[int], cre_start: int, seed: int = 42) -> None:
        self._locked = locked_positions       # TFBS offsets within CRE
        self._cre_start = cre_start           # absolute genomic start of CRE
        self._rng = random.Random(seed)

    def predict_variant_raw(
        self,
        chrom: str,
        interval_start: int,
        interval_end: int,
        variant_pos: int,   # 1-based VCF
        ref: str,
        alt: str,
        ontology_terms: list[str],
        organism: str = "human",
        **kwargs: Any,
    ) -> Any:
        """
        Return synthetic tracks for a single variant.

        The CRE region of the track is set to WT ± effect_delta based on
        whether the mutated position is inside a TFBS.
        """
        is_wt = (ref == alt)
        cre_offset = (variant_pos - 1) - self._cre_start  # 0-based CRE position

        if is_wt:
            atac_mean = self.WT_ATAC
            rna_mean = self.WT_RNA
        elif cre_offset in self._locked:
            # Mutation disrupts a TFBS — large effect
            noise = self._rng.gauss(0, 0.05)
            atac_mean = max(0.0, self.WT_ATAC - 0.30 + noise)
            rna_noise = self._rng.gauss(0, 0.05)
            rna_mean = max(0.0, self.WT_RNA - 0.28 + rna_noise)
        else:
            # Mutation in spacer — small effect
            noise = self._rng.gauss(0, 0.10)
            atac_mean = max(0.0, self.WT_ATAC - 0.05 + noise)
            rna_noise = self._rng.gauss(0, 0.08)
            rna_mean = max(0.0, self.WT_RNA - 0.04 + rna_noise)

        # Build full-window tracks: background signal + CRE peak
        atac_track = np.full(self.TRACK_LENGTH, 0.05, dtype=np.float32)
        rna_track = np.full(self.TRACK_LENGTH, 0.03, dtype=np.float32)

        # Set CRE region signal
        cre_len = 150  # window size
        cre_offset_in_window = self._cre_start - interval_start
        if 0 <= cre_offset_in_window < self.TRACK_LENGTH - cre_len:
            atac_track[cre_offset_in_window: cre_offset_in_window + cre_len] = atac_mean
            rna_track[cre_offset_in_window: cre_offset_in_window + cre_len] = rna_mean

        return _MockOutputs(atac_track, rna_track)

    def extract_track_region(
        self,
        raw_outputs: Any,
        region_start: int,
        region_end: int,
        interval_start: int,
    ) -> tuple[float, float]:
        offset = region_start - interval_start
        length = region_end - region_start
        atac_track: np.ndarray = np.array(raw_outputs.atac_seq)
        rna_track: np.ndarray = np.array(raw_outputs.rna_seq)
        if atac_track.ndim == 1:
            atac_score = float(np.mean(atac_track[offset: offset + length]))
            rna_score = float(np.mean(rna_track[offset: offset + length]))
        else:
            atac_score = float(np.mean(atac_track[:, offset: offset + length]))
            rna_score = float(np.mean(rna_track[:, offset: offset + length]))
        return atac_score, rna_score
