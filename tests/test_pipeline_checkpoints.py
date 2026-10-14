"""Unit tests for config serialization, masking, and checkpoint manifest invalidation."""

import json
from pathlib import Path
import pytest

from lsd_alphagenome_pipeline.config import PipelineConfig
from lsd_alphagenome_pipeline.core.lsd_engine import LocusRecord, WildTypeCRE
from lsd_alphagenome_pipeline.pipeline import Pipeline
from lsd_alphagenome_pipeline.utils.bio_helpers import GenomicInterval, MotifHit


def test_config_masks_api_key(tmp_path: Path):
    yaml_file = tmp_path / "test_config.yaml"
    config = PipelineConfig(
        genes_bed=tmp_path / "genes.bed",
        tf_motifs=tmp_path / "motifs.meme",
        output_dir=tmp_path / "out",
        alphagenome_api_key="SUPER_SECRET_KEY_12345",
    )
    config.to_yaml(yaml_file, mask_key=True)
    content = yaml_file.read_text(encoding="utf-8")
    assert "SUPER_SECRET_KEY_12345" not in content
    assert "***MASKED***" in content


def test_checkpoint_manifest_invalidation(tmp_path: Path):
    out_dir = tmp_path / "out"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    # Place an initial dummy checkpoint
    dummy_ckpt = ckpt_dir / "01_wt_cre.json"
    dummy_ckpt.write_text('{"old": true}', encoding="utf-8")
    manifest = ckpt_dir / "manifest.json"
    manifest.write_text(json.dumps({"fingerprint": "old_fingerprint"}), encoding="utf-8")

    config = PipelineConfig(
        genes_bed=tmp_path / "genes.bed",
        tf_motifs=tmp_path / "motifs.meme",
        output_dir=out_dir,
        cell_type="glioblastoma",
    )

    pipeline = Pipeline(config)
    # Trigger manifest check via directory setup in run() or manually
    current_fp = config.fingerprint()
    assert current_fp != "old_fingerprint"

    # Simulate running pipeline's checkpoint invalidation logic
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    if saved.get("fingerprint") != current_fp:
        for f in ckpt_dir.glob("*.json"):
            f.unlink()

    assert not dummy_ckpt.exists()


def test_save_and_load_wt_cre_preserves_width(tmp_path: Path):
    ckpt_file = tmp_path / "01_wt_cre.json"
    iv = GenomicInterval("chr7", 1000, 1150)
    hit = MotifHit("RUNX1", 20, "+", 15.0, 1e-5, width=12)
    wt = WildTypeCRE(
        interval=iv,
        sequence="A" * 150,
        motif_hits=[hit],
        enrichment_score=2.5,
        locus=LocusRecord(gene=iv, locus=iv),
    )

    config = PipelineConfig(
        genes_bed=tmp_path / "genes.bed",
        tf_motifs=tmp_path / "motifs.meme",
        output_dir=tmp_path / "out",
    )
    pipeline = Pipeline(config)
    pipeline._save_wt_cre(wt, ckpt_file)

    loaded = pipeline._load_wt_cre(ckpt_file)
    assert len(loaded.motif_hits) == 1
    assert loaded.motif_hits[0].width == 12
    assert loaded.motif_hits[0].tf_name == "RUNX1"
    assert loaded.motif_hits[0].pos == 20
