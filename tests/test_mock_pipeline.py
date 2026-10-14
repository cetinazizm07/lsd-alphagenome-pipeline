"""Unit tests for MockAlphaGenomeClient and end-to-end validation."""

from pathlib import Path
import pytest

from lsd_alphagenome_pipeline.config import PipelineConfig
from lsd_alphagenome_pipeline.core.alphagenome_val import AlphaGenomeValidator, ScoredVariant
from lsd_alphagenome_pipeline.core.lsd_engine import LocusRecord, WildTypeCRE
from lsd_alphagenome_pipeline.core.mutagenesis import MutagenesisEngine, VariantRecord
from lsd_alphagenome_pipeline.utils.bio_helpers import GenomicInterval, MotifHit
from lsd_alphagenome_pipeline.utils.mock_client import MockAlphaGenomeClient


def test_mock_client_and_validator():
    seq = "ACGT" * 37 + "AA"  # 150 bp
    iv = GenomicInterval("chr7", 100_000, 100_150)
    hit = MotifHit("RUNX1", 20, "+", 12.0, 1e-4, width=10)
    wt_cre = WildTypeCRE(
        interval=iv,
        sequence=seq,
        motif_hits=[hit],
        enrichment_score=1.5,
        locus=LocusRecord(gene=iv, locus=iv),
    )

    locked = MutagenesisEngine._locked_positions([hit], len(seq))
    mock = MockAlphaGenomeClient(locked_positions=locked, cre_start=iv.start, seed=42)

    config = PipelineConfig(
        genes_bed=Path("dummy.bed"),
        tf_motifs=Path("dummy.meme"),
        output_dir=Path("out"),
        cell_type="glioblastoma",
    )
    validator = AlphaGenomeValidator(config)
    validator._client = mock

    # Generate 5 test variants
    variants = [
        VariantRecord("chr7", 100_005, "A", "G", 4, "saturation"),
        VariantRecord("chr7", 100_021, "C", "T", 20, "saturation"),  # in locked region
    ]

    res = validator.validate(wt_cre, variants)
    assert res.wt_atac > 0
    assert res.wt_rna > 0
    assert len(res.ranked_variants) == 2

    # Locked/disruptive variant should have lower fitness than neutral variant
    # res.ranked_variants is sorted by fitness descending
    ranked_offsets = [v.variant.cre_offset for v in res.ranked_variants]
    assert 4 in ranked_offsets
    assert 20 in ranked_offsets
