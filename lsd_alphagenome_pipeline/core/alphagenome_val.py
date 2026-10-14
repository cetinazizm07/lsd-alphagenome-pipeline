"""
AlphaGenome batch validation module.

Predicts ATAC-seq and RNA-seq tracks for the WT CRE and each variant,
then computes log2 fold-change and harmonic-mean fitness scores.

Architecture notes
------------------
AlphaGenome's predict_variant() operates on a ~1 Mbp genomic window and
returns per-base signal tracks.  We must:
  1. Build a 1 Mbp Interval centred on the CRE midpoint.
  2. Submit each VariantRecord as a VCF-style genome.Variant (1-based pos).
  3. Extract the CRE sub-region from the returned track (region aggregation).
  4. Compare each mutant's aggregated score to the WT baseline.

Ref: AlphaGenome README §Quick start
     https://github.com/google-deepmind/alphagenome

Fitness formula (harmonic mean — LSD paper §Methods "Fitness scoring"):
    Fitness = (2 × ATAC × RNA) / (ATAC + RNA + ε)

Log2FC:
    log2FC_atac = log2(mutant_atac / (wt_atac + ε))
    log2FC_rna  = log2(mutant_rna  / (wt_rna  + ε))
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from lsd_alphagenome_pipeline.config import PipelineConfig
from lsd_alphagenome_pipeline.core.lsd_engine import WildTypeCRE
from lsd_alphagenome_pipeline.core.mutagenesis import VariantRecord
from lsd_alphagenome_pipeline.utils.api_client import AlphaGenomeClient, VariantPrediction

log = logging.getLogger(__name__)

_EPS = 1e-9  # numerical guard


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ScoredVariant:
    """Final output row: variant identity + AlphaGenome predictions + fitness."""
    variant: VariantRecord
    atac_score: float
    rna_score: float
    log2fc_atac: float
    log2fc_rna: float
    fitness: float      # harmonic mean of ATAC + RNA signals

    def as_dict(self) -> dict:
        return {
            "chrom": self.variant.chrom,
            "pos_1based": self.variant.pos,
            "ref": self.variant.ref,
            "alt": self.variant.alt,
            "cre_offset": self.variant.cre_offset,
            "strategy": self.variant.strategy,
            "atac_score": self.atac_score,
            "rna_score": self.rna_score,
            "log2fc_atac": self.log2fc_atac,
            "log2fc_rna": self.log2fc_rna,
            "fitness": self.fitness,
        }


@dataclass
class ValidationResult:
    """Container for the full WT + mutant comparison."""
    wt_atac: float
    wt_rna: float
    wt_fitness: float
    ranked_variants: list[ScoredVariant]  # sorted by fitness descending


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class AlphaGenomeValidator:
    """
    Orchestrates batch AlphaGenome predictions and fitness ranking.

    Usage
    -----
    validator = AlphaGenomeValidator(config)
    result = validator.validate(wt_cre, variants)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        if config.alphagenome_api_key == "mock":
            from lsd_alphagenome_pipeline.utils.mock_client import MockAlphaGenomeClient
            self._client = MockAlphaGenomeClient(
                locked_positions=frozenset(),
                cre_start=0,
                seed=config.random_seed,
            )
        else:
            self._client = AlphaGenomeClient(
                api_key=config.alphagenome_api_key,
                max_retries=config.alphagenome_max_retries,
                backoff_base=config.alphagenome_backoff_base,
            )
        self._ontology = [config.cell_ontology_id()]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def validate(
        self,
        wt_cre: WildTypeCRE,
        variants: list[VariantRecord],
    ) -> ValidationResult:
        """
        Run WT prediction then all variant predictions.

        WT is predicted as the reference allele at the first variant's
        position — this gives a consistent baseline using the same
        AlphaGenome window as the mutants.
        """
        if self.config.alphagenome_api_key == "mock":
            from lsd_alphagenome_pipeline.core.mutagenesis import MutagenesisEngine
            from lsd_alphagenome_pipeline.utils.mock_client import MockAlphaGenomeClient
            locked = MutagenesisEngine._locked_positions(wt_cre.motif_hits, len(wt_cre.sequence))
            self._client = MockAlphaGenomeClient(
                locked_positions=locked,
                cre_start=wt_cre.interval.start,
                seed=self.config.random_seed,
            )
        interval_start, interval_end = self._build_context_interval(wt_cre)
        log.info(
            "AlphaGenome context window: %s:%d-%d (%d bp)",
            wt_cre.interval.chrom, interval_start, interval_end,
            interval_end - interval_start
        )

        log.info("Predicting WT baseline ...")
        wt_atac, wt_rna = self._predict_wt(wt_cre, interval_start, interval_end)
        wt_fitness = self._harmonic_mean(wt_atac, wt_rna)
        log.info("WT: atac=%.6f  rna=%.6f  fitness=%.6f", wt_atac, wt_rna, wt_fitness)

        log.info("Predicting %d variants in batches of %d ...", len(variants), self.config.alphagenome_batch_size)
        scored: list[ScoredVariant] = []
        for batch_start in range(0, len(variants), self.config.alphagenome_batch_size):
            batch = variants[batch_start: batch_start + self.config.alphagenome_batch_size]
            batch_scored = self._predict_batch(
                batch, wt_cre, interval_start, interval_end, wt_atac, wt_rna
            )
            scored.extend(batch_scored)
            log.info(
                "Completed %d / %d variants",
                min(batch_start + self.config.alphagenome_batch_size, len(variants)),
                len(variants),
            )

        scored.sort(key=lambda s: s.fitness, reverse=True)
        return ValidationResult(
            wt_atac=wt_atac,
            wt_rna=wt_rna,
            wt_fitness=wt_fitness,
            ranked_variants=scored,
        )

    # ------------------------------------------------------------------
    # Internal prediction methods
    # ------------------------------------------------------------------

    def _build_context_interval(self, wt_cre: WildTypeCRE) -> tuple[int, int]:
        """
        Build a 1 Mbp window centred on the CRE midpoint.

        AlphaGenome requires exactly 2^20 = 1,048,576 bp input.
        We centre on the CRE midpoint and clip to chromosome bounds (assuming
        chromosome length >> 1 Mbp; for edge cases the API will return an error).

        Ref: AlphaGenome README §Interval — "The interval must be exactly
        1,048,576 bases" (note: verify against actual installed package docs).
        """
        ctx = self.config.alphagenome_context_bp  # 1_048_576
        mid = wt_cre.interval.midpoint
        start = max(0, mid - ctx // 2)
        end = start + ctx
        return start, end

    def _predict_wt(
        self,
        wt_cre: WildTypeCRE,
        interval_start: int,
        interval_end: int,
    ) -> tuple[float, float]:
        """
        Predict the reference allele (WT) by passing ref=alt for the first
        mutable position, which forces AlphaGenome to return the unmodified
        track for the context window.

        If no variants exist (fully locked CRE), use position 1 of CRE + ref.
        """
        seq = wt_cre.sequence
        # Any position works for WT; pick the CRE midpoint.
        cre_mid_offset = len(seq) // 2
        ref_base = seq[cre_mid_offset]
        abs_pos_1based = wt_cre.interval.start + cre_mid_offset + 1

        organism = "mouse" if self.config.assembly in ("mm10", "mm39") else "human"
        raw = self._client.predict_variant_raw(
            chrom=wt_cre.interval.chrom,
            interval_start=interval_start,
            interval_end=interval_end,
            variant_pos=abs_pos_1based,
            ref=ref_base,
            alt=ref_base,   # ref == alt → no actual mutation, returns WT track
            ontology_terms=self._ontology,
            organism=organism,
        )
        return self._aggregate(raw, wt_cre, interval_start)

    def _predict_batch(
        self,
        batch: list[VariantRecord],
        wt_cre: WildTypeCRE,
        interval_start: int,
        interval_end: int,
        wt_atac: float,
        wt_rna: float,
    ) -> list[ScoredVariant]:
        scored: list[ScoredVariant] = []
        organism = "mouse" if self.config.assembly in ("mm10", "mm39") else "human"
        for variant in batch:
            try:
                raw = self._client.predict_variant_raw(
                    chrom=variant.chrom,
                    interval_start=interval_start,
                    interval_end=interval_end,
                    variant_pos=variant.pos,
                    ref=variant.ref,
                    alt=variant.alt,
                    ontology_terms=self._ontology,
                    organism=organism,
                )
                atac, rna = self._aggregate(raw, wt_cre, interval_start)
            except Exception as exc:
                log.error(
                    "Prediction failed for variant %s:%d%s>%s — %s",
                    variant.chrom, variant.pos, variant.ref, variant.alt, exc
                )
                continue

            eps = self.config.fitness_epsilon
            log2fc_atac = math.log2((atac + eps) / (wt_atac + eps))
            log2fc_rna = math.log2((rna + eps) / (wt_rna + eps))
            fitness = self._harmonic_mean(atac, rna, eps)

            scored.append(ScoredVariant(
                variant=variant,
                atac_score=atac,
                rna_score=rna,
                log2fc_atac=log2fc_atac,
                log2fc_rna=log2fc_rna,
                fitness=fitness,
            ))
        return scored

    def _aggregate(
        self,
        raw_outputs,
        wt_cre: WildTypeCRE,
        interval_start: int,
    ) -> tuple[float, float]:
        """
        Extract aggregated (atac, rna) scalars from the AlphaGenome output.

        Aggregation region is determined by config.track_agg_region:
          "cre"        — mean over the exact CRE genomic span
          "tss_window" — mean over ±tss_window_bp from the gene TSS (or CRE midpoint fallback)
        """
        if self.config.track_agg_region == "cre":
            region_start = wt_cre.interval.start
            region_end = wt_cre.interval.end
        else:
            if hasattr(wt_cre, "locus") and wt_cre.locus and hasattr(wt_cre.locus, "gene") and wt_cre.locus.gene:
                g = wt_cre.locus.gene
                mid = g.end if g.strand == "-" else g.start
            else:
                mid = wt_cre.interval.midpoint
            hw = self.config.tss_window_bp
            region_start = max(0, mid - hw)
            region_end = mid + hw

        return self._client.extract_track_region(
            raw_outputs=raw_outputs,
            region_start=region_start,
            region_end=region_end,
            interval_start=interval_start,
        )

    @staticmethod
    def _harmonic_mean(atac: float, rna: float, eps: float = _EPS) -> float:
        """
        Compute harmonic mean fitness score.

        Fitness = (2 × ATAC × RNA) / (ATAC + RNA + ε)

        The ε guard prevents division by zero when both tracks are zero.
        Using harmonic mean (vs arithmetic) penalises cases where one assay
        is high but the other is near zero — this selects for balanced
        chromatin accessibility + transcriptional activity.
        """
        return (2.0 * atac * rna) / (atac + rna + eps)
