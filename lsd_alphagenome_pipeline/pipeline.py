"""
Full pipeline orchestration: LSD → Mutagenesis → AlphaGenome → Export.

Checkpointing
-------------
Each step writes an intermediate checkpoint file so a crashed run can
resume from the last completed step rather than starting over.  Checkpoint
files are JSON and named:
  output_dir/checkpoints/01_wt_cre.json
  output_dir/checkpoints/02_variants.json
  output_dir/checkpoints/03_validation.json

The pipeline detects existing checkpoints at startup and skips completed steps.

Export format
-------------
output_dir/results.csv       — ranked variants (config.top_n_export rows)
output_dir/results.json      — same data in JSON
output_dir/run_config.yaml   — exact config used (for reproducibility)
output_dir/wt_cre.fa         — WT CRE FASTA sequence
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from lsd_alphagenome_pipeline.config import PipelineConfig
from lsd_alphagenome_pipeline.core.alphagenome_val import (
    AlphaGenomeValidator,
    ScoredVariant,
    ValidationResult,
)
from lsd_alphagenome_pipeline.core.lsd_engine import LSDEngine, WildTypeCRE
from lsd_alphagenome_pipeline.core.mutagenesis import MutagenesisEngine, VariantRecord
from lsd_alphagenome_pipeline.utils.bio_helpers import GenomicInterval

log = logging.getLogger(__name__)


class Pipeline:
    """
    Top-level orchestrator.

    Usage
    -----
    pipeline = Pipeline(config)
    pipeline.run()
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._ckpt_dir = config.output_dir / "checkpoints"

    def run(self) -> Path:
        """
        Execute all pipeline steps with checkpointing.

        Returns path to the output CSV.
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._ckpt_dir.mkdir(exist_ok=True)

        # Persist config for reproducibility — written before any computation
        config_path = self.config.output_dir / "run_config.yaml"
        self.config.to_yaml(config_path)
        log.info("Run config saved to %s", config_path)

        # ------------------------------------------------------------------
        # Step 1: LSD CRE selection
        # ------------------------------------------------------------------
        wt_cre = self._step1_lsd()

        # ------------------------------------------------------------------
        # Step 2: Mutagenesis
        # ------------------------------------------------------------------
        variants = self._step2_mutagenesis(wt_cre)

        # ------------------------------------------------------------------
        # Step 3: AlphaGenome validation
        # ------------------------------------------------------------------
        result = self._step3_validation(wt_cre, variants)

        # ------------------------------------------------------------------
        # Step 4: Export
        # ------------------------------------------------------------------
        csv_path = self._export(wt_cre, result)
        return csv_path

    # ------------------------------------------------------------------
    # Checkpointed steps
    # ------------------------------------------------------------------

    def _step1_lsd(self) -> WildTypeCRE:
        ckpt = self._ckpt_dir / "01_wt_cre.json"
        if ckpt.exists():
            log.info("Checkpoint found — skipping LSD step, loading %s", ckpt)
            return self._load_wt_cre(ckpt)

        t0 = time.monotonic()
        engine = LSDEngine(self.config)
        wt_cre = engine.select_wt_cre()
        self._save_wt_cre(wt_cre, ckpt)
        log.info("LSD step completed in %.1f s", time.monotonic() - t0)
        return wt_cre

    def _step2_mutagenesis(self, wt_cre: WildTypeCRE) -> list[VariantRecord]:
        ckpt = self._ckpt_dir / "02_variants.json"
        if ckpt.exists():
            log.info("Checkpoint found — skipping mutagenesis step, loading %s", ckpt)
            return self._load_variants(ckpt)

        t0 = time.monotonic()
        engine = MutagenesisEngine(self.config)
        variants = engine.generate(wt_cre)
        self._save_variants(variants, ckpt)
        log.info("Mutagenesis step completed in %.1f s — %d variants", time.monotonic() - t0, len(variants))
        return variants

    def _step3_validation(
        self,
        wt_cre: WildTypeCRE,
        variants: list[VariantRecord],
    ) -> ValidationResult:
        ckpt = self._ckpt_dir / "03_validation.json"
        if ckpt.exists():
            log.info("Checkpoint found — skipping validation step, loading %s", ckpt)
            return self._load_validation(ckpt)

        t0 = time.monotonic()
        validator = AlphaGenomeValidator(self.config)
        result = validator.validate(wt_cre, variants)
        self._save_validation(result, ckpt)
        log.info("Validation step completed in %.1f s", time.monotonic() - t0)
        return result

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(self, wt_cre: WildTypeCRE, result: ValidationResult) -> Path:
        out = self.config.output_dir
        n = self.config.top_n_export
        top = result.ranked_variants[:n]

        # CSV
        rows = [s.as_dict() for s in top]
        df = pd.DataFrame(rows)
        csv_path = out / "results.csv"
        df.to_csv(csv_path, index=False)
        log.info("Exported top %d variants to %s", n, csv_path)

        # JSON
        json_path = out / "results.json"
        payload = {
            "wt": {
                "chrom": wt_cre.interval.chrom,
                "start": wt_cre.interval.start,
                "end": wt_cre.interval.end,
                "atac_score": result.wt_atac,
                "rna_score": result.wt_rna,
                "fitness": result.wt_fitness,
            },
            "top_variants": rows,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # WT FASTA
        fa_path = out / "wt_cre.fa"
        fa_path.write_text(
            f">{wt_cre.interval.chrom}:{wt_cre.interval.start}-{wt_cre.interval.end}\n"
            f"{wt_cre.sequence}\n",
            encoding="utf-8",
        )
        log.info("WT CRE sequence written to %s", fa_path)

        return csv_path

    # ------------------------------------------------------------------
    # Checkpoint serialisation helpers
    # ------------------------------------------------------------------

    def _save_wt_cre(self, wt: WildTypeCRE, path: Path) -> None:
        data = {
            "interval": {
                "chrom": wt.interval.chrom,
                "start": wt.interval.start,
                "end": wt.interval.end,
                "name": wt.interval.name,
                "strand": wt.interval.strand,
            },
            "sequence": wt.sequence,
            "enrichment_score": wt.enrichment_score,
            "motif_hits": [
                {
                    "tf_name": h.tf_name,
                    "pos": h.pos,
                    "strand": h.strand,
                    "score": h.score,
                    "pvalue": h.pvalue,
                }
                for h in wt.motif_hits
            ],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_wt_cre(self, path: Path) -> WildTypeCRE:
        from lsd_alphagenome_pipeline.utils.bio_helpers import MotifHit
        from lsd_alphagenome_pipeline.core.lsd_engine import LocusRecord
        data = json.loads(path.read_text(encoding="utf-8"))
        iv = data["interval"]
        interval = GenomicInterval(iv["chrom"], iv["start"], iv["end"], iv.get("name", ""), iv.get("strand", "."))
        hits = [
            MotifHit(h["tf_name"], h["pos"], h["strand"], h["score"], h["pvalue"])
            for h in data.get("motif_hits", [])
        ]
        # LocusRecord not stored in checkpoint (not needed for downstream steps)
        dummy_locus = LocusRecord(gene=interval, locus=interval)
        return WildTypeCRE(
            interval=interval,
            sequence=data["sequence"],
            motif_hits=hits,
            enrichment_score=data["enrichment_score"],
            locus=dummy_locus,
        )

    def _save_variants(self, variants: list[VariantRecord], path: Path) -> None:
        data = [
            {
                "chrom": v.chrom,
                "pos": v.pos,
                "ref": v.ref,
                "alt": v.alt,
                "cre_offset": v.cre_offset,
                "strategy": v.strategy,
            }
            for v in variants
        ]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_variants(self, path: Path) -> list[VariantRecord]:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            VariantRecord(
                chrom=d["chrom"],
                pos=d["pos"],
                ref=d["ref"],
                alt=d["alt"],
                cre_offset=d["cre_offset"],
                strategy=d["strategy"],
            )
            for d in data
        ]

    def _save_validation(self, result: ValidationResult, path: Path) -> None:
        data = {
            "wt_atac": result.wt_atac,
            "wt_rna": result.wt_rna,
            "wt_fitness": result.wt_fitness,
            "ranked_variants": [s.as_dict() for s in result.ranked_variants],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_validation(self, path: Path) -> ValidationResult:
        from lsd_alphagenome_pipeline.core.mutagenesis import VariantRecord
        data = json.loads(path.read_text(encoding="utf-8"))
        variants = []
        for d in data["ranked_variants"]:
            vr = VariantRecord(
                chrom=d["chrom"],
                pos=d["pos_1based"],
                ref=d["ref"],
                alt=d["alt"],
                cre_offset=d["cre_offset"],
                strategy=d["strategy"],
            )
            variants.append(ScoredVariant(
                variant=vr,
                atac_score=d["atac_score"],
                rna_score=d["rna_score"],
                log2fc_atac=d["log2fc_atac"],
                log2fc_rna=d["log2fc_rna"],
                fitness=d["fitness"],
            ))
        return ValidationResult(
            wt_atac=data["wt_atac"],
            wt_rna=data["wt_rna"],
            wt_fitness=data["wt_fitness"],
            ranked_variants=variants,
        )
