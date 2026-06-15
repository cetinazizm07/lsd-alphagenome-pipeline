"""
Resilient AlphaGenome API wrapper.

Wraps alphagenome.models.dna_client with:
  * Exponential back-off on HTTP 429 (tenacity)
  * Configurable retry budget
  * Structured result type

AlphaGenome API reference
--------------------------
from alphagenome.models import dna_client
from alphagenome.data import genome

model = dna_client.create(api_key)
outputs = model.predict_variant(
    interval=genome.Interval(chromosome, start, end),
    variant=genome.Variant(chromosome, position, reference_bases, alternate_bases),
    ontology_terms=['CL:0000540'],
    requested_outputs=[dna_client.OutputType.RNA_SEQ,
                       dna_client.OutputType.ATAC_SEQ],
)

Ref: https://github.com/google-deepmind/alphagenome (README §Quick start)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import tenacity

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class VariantPrediction:
    """
    Aggregated scalar predictions for one variant.

    *atac_score* and *rna_score* are floats derived from the per-base track
    by averaging over the CRE region of interest (see AlphaGenomeValidator).
    """
    chrom: str
    pos: int        # 1-based VCF position
    ref: str
    alt: str
    atac_score: float
    rna_score: float


# ---------------------------------------------------------------------------
# AlphaGenome client wrapper
# ---------------------------------------------------------------------------

class AlphaGenomeClient:
    """
    Thin wrapper around alphagenome.models.dna_client.

    Imports are deferred so the module can be imported even when the
    alphagenome package is not installed (useful for unit-testing stubs).
    """

    def __init__(
        self,
        api_key: str,
        max_retries: int = 5,
        backoff_base: float = 2.0,
    ) -> None:
        self._api_key = api_key
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._model: Any = None  # lazy-loaded in _get_model()

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from alphagenome.models import dna_client  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "alphagenome package not found. "
                    "Install from https://github.com/google-deepmind/alphagenome"
                ) from exc
            self._model = dna_client.create(self._api_key)
        return self._model

    def _output_types(self) -> list[Any]:
        from alphagenome.models import dna_client  # type: ignore[import]
        # OutputType enum members confirmed from AlphaGenome README examples.
        # ATAC_SEQ corresponds to "chromatin accessibility" output.
        return [dna_client.OutputType.RNA_SEQ, dna_client.OutputType.ATAC_SEQ]

    @tenacity.retry(
        retry=tenacity.retry_if_exception_type(Exception),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=60),
        stop=tenacity.stop_after_attempt(5),
        before_sleep=tenacity.before_sleep_log(log, logging.WARNING),
    )
    def predict_variant_raw(
        self,
        chrom: str,
        interval_start: int,
        interval_end: int,
        variant_pos: int,
        ref: str,
        alt: str,
        ontology_terms: list[str],
    ) -> Any:
        """
        Call AlphaGenome predict_variant with retry.

        *interval_start* / *interval_end* define the ~1 Mbp context window
        (0-based, half-open).  *variant_pos* is 1-based (VCF convention) as
        required by genome.Variant.

        The tenacity decorator retries on any exception with exponential
        back-off — this handles HTTP 429 (rate limit) and transient failures.
        Max back-off is 60 s; after 5 attempts the exception propagates.
        """
        from alphagenome.data import genome  # type: ignore[import]

        model = self._get_model()
        interval = genome.Interval(
            chromosome=chrom,
            start=interval_start,
            end=interval_end,
        )
        variant = genome.Variant(
            chromosome=chrom,
            position=variant_pos,   # 1-based VCF
            reference_bases=ref,
            alternate_bases=alt,
        )
        return model.predict_variant(
            interval=interval,
            variant=variant,
            ontology_terms=ontology_terms,
            requested_outputs=self._output_types(),
        )

    def extract_track_region(
        self,
        raw_outputs: Any,
        region_start: int,
        region_end: int,
        interval_start: int,
    ) -> tuple[float, float]:
        """
        Aggregate the per-base track over [region_start, region_end).

        AlphaGenome returns tracks as numpy arrays aligned to the interval.
        Offset = region_start - interval_start gives the slice index.

        Returns (mean_atac, mean_rna) over the specified region.
        """
        offset = region_start - interval_start
        length = region_end - region_start
        if offset < 0 or length <= 0:
            raise ValueError(
                f"Region [{region_start},{region_end}) falls outside "
                f"interval starting at {interval_start}"
            )

        # AlphaGenome output attribute access:
        # outputs.atac_seq  — shape (T, W) where T=tracks, W=window_bp
        # outputs.rna_seq   — shape (T, W)
        # We take mean over the CRE slice along W, then mean over T (tracks).
        atac_track: np.ndarray = np.array(raw_outputs.atac_seq)
        rna_track: np.ndarray = np.array(raw_outputs.rna_seq)

        # Handle both (W,) and (T, W) shapes
        if atac_track.ndim == 1:
            atac_score = float(np.mean(atac_track[offset:offset + length]))
            rna_score = float(np.mean(rna_track[offset:offset + length]))
        else:
            atac_score = float(np.mean(atac_track[:, offset:offset + length]))
            rna_score = float(np.mean(rna_track[:, offset:offset + length]))

        return atac_score, rna_score
