"""Unit tests for bio_helpers module."""

from pathlib import Path
import tempfile
import pytest

from lsd_alphagenome_pipeline.utils.bio_helpers import (
    GenomicInterval,
    MotifHit,
    motif_hit_to_absolute,
    parse_bed,
)


def test_genomic_interval_properties():
    iv = GenomicInterval("chr7", 100, 250, "test_cre", "+")
    assert iv.length == 150
    assert iv.midpoint == 175

    expanded = iv.expand(50)
    assert expanded.start == 50
    assert expanded.end == 300
    assert expanded.chrom == "chr7"


def test_motif_hit_and_absolute():
    hit = MotifHit(tf_name="RUNX1", pos=15, strand="+", score=12.5, pvalue=1e-5, width=11)
    assert hit.width == 11
    assert hit.pos == 15

    locus = GenomicInterval("chr7", 1000, 2000)
    abs_start, abs_end = motif_hit_to_absolute(hit, locus)
    assert abs_start == 1015
    assert abs_end == 1026


def test_parse_bed(tmp_path: Path):
    bed_file = tmp_path / "test.bed"
    bed_file.write_text(
        "# comment\n"
        "chr7\t1000\t2000\tGENE1\t100\t+\n"
        "chr7\t5000\t6000\tGENE2\t200\t-\n",
        encoding="utf-8",
    )
    intervals = parse_bed(bed_file, min_cols=3)
    assert len(intervals) == 2
    assert intervals[0].chrom == "chr7"
    assert intervals[0].start == 1000
    assert intervals[0].end == 2000
    assert intervals[0].name == "GENE1"
    assert intervals[0].strand == "+"
    assert intervals[1].strand == "-"
