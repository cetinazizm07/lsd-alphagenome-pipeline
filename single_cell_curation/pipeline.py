"""
CurationPipeline — checkpointed orchestrator for single-cell and bulk curation.

Checkpoints
-----------
Each step writes <output_dir>/checkpoints/<NN>_<step>.json containing
{"h5ad_path": "...", "n_obs": N, "n_vars": M}.  On resume the pipeline
loads the H5AD pointer from the JSON and skips recomputation.

Single-cell path:  ingest → qc_metrics → doublets → ambient → process → export
Bulk path:         ingest → bulk_normalize → export
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import anndata

from single_cell_curation.config import CurationConfig
from single_cell_curation.export.report import write_report
from single_cell_curation.export.writers import write_outputs
from single_cell_curation.ingest import load_dataset
from single_cell_curation.processing.bulk_processing import run_bulk_pipeline
from single_cell_curation.processing.sc_processing import run_sc_pipeline
from single_cell_curation.qc.ambient import run_soupx
from single_cell_curation.qc.cell_cycle import run_cell_cycle
from single_cell_curation.qc.doublets import run_scrublet
from single_cell_curation.qc.metrics import run_qc_metrics
from single_cell_curation.utils.io_helpers import (
    load_adata_from_checkpoint,
    save_checkpoint,
)

log = logging.getLogger(__name__)


@dataclass
class CurationResult:
    adata: anndata.AnnData
    report: dict
    output_dir: Path


class CurationPipeline:
    """Top-level pipeline orchestrator with per-step checkpointing.

    Usage
    -----
    config = CurationConfig(dataset_name="Darmanis_2017", ...)
    result = CurationPipeline(config).run()
    result.adata        # fully processed AnnData
    result.output_dir   # where files were written
    """

    def __init__(self, config: CurationConfig) -> None:
        self.config = config
        self._ckpt_dir = config.output_dir / "checkpoints"

    def run(self) -> CurationResult:
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self._ckpt_dir.mkdir(exist_ok=True)

        cfg.to_yaml(cfg.output_dir / "run_config.yaml")

        if cfg.modality == "single_cell":
            adata = self._run_sc()
        else:
            adata = self._run_bulk()

        report_path = write_report(adata, cfg)
        import json
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return CurationResult(adata=adata, report=report, output_dir=cfg.output_dir)

    # ------------------------------------------------------------------
    # Single-cell path
    # ------------------------------------------------------------------

    def _run_sc(self) -> anndata.AnnData:
        adata = self._checkpointed("01_ingest", lambda: load_dataset(self.config))
        adata = self._checkpointed("02_qc_metrics", lambda: run_qc_metrics(adata, self.config))
        adata = self._checkpointed("03_doublets", lambda: run_scrublet(adata, self.config))
        adata = self._checkpointed("04_ambient", lambda: run_soupx(adata, self.config))
        adata = self._checkpointed("05_process", lambda: self._sc_process(adata))
        h5ad_path = write_outputs(adata, self.config)
        self._mark_done("06_export", adata, h5ad_path)
        return adata

    def _sc_process(self, adata: anndata.AnnData) -> anndata.AnnData:
        adata = run_sc_pipeline(adata, self.config)
        if self.config.score_cell_cycle:
            adata = run_cell_cycle(adata, self.config)
        return adata

    # ------------------------------------------------------------------
    # Bulk path
    # ------------------------------------------------------------------

    def _run_bulk(self) -> anndata.AnnData:
        adata = self._checkpointed("01_ingest", lambda: load_dataset(self.config))

        def _bulk_normalize() -> anndata.AnnData:
            import pandas as pd
            import scipy.sparse as sp
            counts_df = pd.DataFrame(
                adata.X.toarray() if sp.issparse(adata.X) else adata.X,
                index=adata.obs_names,
                columns=adata.var_names,
            ).T
            return run_bulk_pipeline(counts_df, adata.obs.copy(), self.config)

        adata = self._checkpointed("02_bulk_normalize", _bulk_normalize)
        h5ad_path = write_outputs(adata, self.config)
        self._mark_done("03_export", adata, h5ad_path)
        return adata

    # ------------------------------------------------------------------
    # Checkpointing helpers
    # ------------------------------------------------------------------

    def _checkpointed(self, step_name: str, fn) -> anndata.AnnData:
        ckpt = self._ckpt_dir / f"{step_name}.json"
        if ckpt.exists():
            log.info("Checkpoint found — skipping %s", step_name)
            return load_adata_from_checkpoint(ckpt)

        t0 = time.monotonic()
        result: anndata.AnnData = fn()
        h5ad_path = self._ckpt_dir / f"{step_name}.h5ad"
        result.write_h5ad(h5ad_path)
        save_checkpoint(ckpt, h5ad_path, {"n_obs": result.n_obs, "n_vars": result.n_vars})
        log.info("Step %s done in %.1f s (%d × %d)", step_name,
                 time.monotonic() - t0, result.n_obs, result.n_vars)
        return result

    def _mark_done(self, step_name: str, adata: anndata.AnnData, h5ad_path: Path) -> None:
        ckpt = self._ckpt_dir / f"{step_name}.json"
        if not ckpt.exists():
            save_checkpoint(ckpt, h5ad_path, {"n_obs": adata.n_obs, "n_vars": adata.n_vars})
