"""Unit tests for mutagenesis engine: locked positions and spacer shuffling."""

from pathlib import Path
import pytest

from lsd_alphagenome_pipeline.config import PipelineConfig
from lsd_alphagenome_pipeline.core.lsd_engine import LocusRecord, WildTypeCRE
from lsd_alphagenome_pipeline.core.mutagenesis import MutagenesisEngine, VariantRecord
from lsd_alphagenome_pipeline.utils.bio_helpers import GenomicInterval, MotifHit


def make_dummy_wt_cre(hits: list[MotifHit], seq: str) -> WildTypeCRE:
    iv = GenomicInterval("chr7", 1000, 1000 + len(seq))
    locus = LocusRecord(gene=iv, locus=iv)
    return WildTypeCRE(
        interval=iv,
        sequence=seq,
        motif_hits=hits,
        enrichment_score=1.0,
        locus=locus,
    )


def test_locked_positions_with_widths():
    # Hit at pos 10 with width 12 -> positions 10..21 locked
    # Hit at pos 50 with width 10 -> positions 50..59 locked
    hits = [
        MotifHit("TF1", 10, "+", 10.0, 1e-4, width=12),
        MotifHit("TF2", 50, "+", 10.0, 1e-4, width=10),
    ]
    locked = MutagenesisEngine._locked_positions(hits, 150)
    assert 10 in locked
    assert 21 in locked
    assert 22 not in locked
    assert 50 in locked
    assert 59 in locked
    assert 60 not in locked
    assert len(locked) == 22


def test_saturation_skips_locked_positions():
    seq = "A" * 150
    hits = [
        MotifHit("TF1", 10, "+", 10.0, 1e-4, width=10),  # locks [10, 20)
    ]
    wt_cre = make_dummy_wt_cre(hits, seq)
    config = PipelineConfig(
        genes_bed=Path("dummy.bed"),
        tf_motifs=Path("dummy.meme"),
        output_dir=Path("out"),
        strategy="saturation",
        max_variants=1000,
    )
    engine = MutagenesisEngine(config)
    variants = engine.generate(wt_cre)

    # 150 - 10 locked = 140 mutable bases
    # 140 * 3 alts = 420 variants
    assert len(variants) == 420

    # Ensure no variant was placed in locked region [10, 20)
    offsets = {v.cre_offset for v in variants}
    for offset in range(10, 20):
        assert offset not in offsets


def test_identify_spacers_and_merging():
    # Overlapping hits: [10, 25) and [20, 35) -> merged span [10, 35)
    # Next hit: [50, 60)
    # Spacer should be [35, 50)
    hits = [
        MotifHit("TF1", 10, "+", 10.0, 1e-4, width=15),
        MotifHit("TF2", 20, "+", 10.0, 1e-4, width=15),
        MotifHit("TF3", 50, "+", 10.0, 1e-4, width=10),
    ]
    spacers = MutagenesisEngine._identify_spacers(hits, 150)
    assert len(spacers) == 1
    assert spacers[0] == (35, 50)


def test_spacer_shuffling_does_not_crash_or_duplicate():
    seq = "ACGT" * 30  # 120 bp
    hits = [
        MotifHit("TF1", 10, "+", 10.0, 1e-4, width=10),
        MotifHit("TF2", 60, "+", 10.0, 1e-4, width=10),
    ]
    wt_cre = make_dummy_wt_cre(hits, seq)
    config = PipelineConfig(
        genes_bed=Path("dummy.bed"),
        tf_motifs=Path("dummy.meme"),
        output_dir=Path("out"),
        strategy="spacer",
        spacer_max_delta_bp=3,
    )
    engine = MutagenesisEngine(config)
    variants = engine.generate(wt_cre)

    assert len(variants) > 0
    # Check that all variants are valid offsets
    for v in variants:
        assert 0 <= v.cre_offset < len(seq)
        assert v.strategy == "spacer"
        assert v.ref != v.alt

    # Check deduplication
    tuples = [(v.cre_offset, v.alt) for v in variants]
    assert len(tuples) == len(set(tuples))
