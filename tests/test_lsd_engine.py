"""Unit tests for LSD engine and enrichment scoring."""

from lsd_alphagenome_pipeline.config import PipelineConfig
from lsd_alphagenome_pipeline.core.lsd_engine import LSDEngine
from lsd_alphagenome_pipeline.utils.bio_helpers import GenomicInterval, MotifHit


def test_bound_by_ctcf():
    config = PipelineConfig(
        genes_bed="dummy.bed",  # type: ignore
        tf_motifs="dummy.meme",  # type: ignore
        output_dir="dummy_out",  # type: ignore
        locus_flank_bp=50_000,
    )
    engine = LSDEngine(config)

    gene = GenomicInterval("chr7", 100_000, 110_000, "TARGET_GENE")
    ctcf_sites = [
        GenomicInterval("chr7", 70_000, 70_500, "CTCF_UP"),
        GenomicInterval("chr7", 140_000, 140_500, "CTCF_DOWN"),
        GenomicInterval("chr1", 50_000, 50_500, "CTCF_OTHER"),
    ]

    locus = engine._bound_by_ctcf(gene, ctcf_sites)
    assert locus.chrom == "chr7"
    assert locus.start == 70_000
    assert locus.end == 140_500


def test_bound_by_ctcf_fallback_when_none():
    config = PipelineConfig(
        genes_bed="dummy.bed",  # type: ignore
        tf_motifs="dummy.meme",  # type: ignore
        output_dir="dummy_out",  # type: ignore
        locus_flank_bp=10_000,
    )
    engine = LSDEngine(config)

    gene = GenomicInterval("chr7", 50_000, 60_000, "TARGET_GENE")
    locus = engine._bound_by_ctcf(gene, [])
    assert locus.chrom == "chr7"
    assert locus.start == 40_000
    assert locus.end == 70_000


def test_enrichment_score_rewards_tf_diversity():
    # Window with 3 different TFs
    diverse_hits = [
        MotifHit("RUNX1", 10, "+", 10.0, 1e-4, width=10),
        MotifHit("NFKB1", 40, "+", 10.0, 1e-4, width=10),
        MotifHit("SRF", 80, "+", 10.0, 1e-4, width=10),
    ]

    # Window with 3 repetitive hits of the same TF
    repetitive_hits = [
        MotifHit("RUNX1", 10, "+", 10.0, 1e-4, width=10),
        MotifHit("RUNX1", 40, "+", 10.0, 1e-4, width=10),
        MotifHit("RUNX1", 80, "+", 10.0, 1e-4, width=10),
    ]

    score_diverse = LSDEngine._enrichment_score(diverse_hits, 150)
    score_repetitive = LSDEngine._enrichment_score(repetitive_hits, 150)

    # Diverse TF representation must strictly outscore repetitive single-TF window
    assert score_diverse > score_repetitive
