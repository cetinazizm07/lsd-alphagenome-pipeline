from __future__ import annotations

import logging

import anndata
import numpy as np

from single_cell_curation.config import CurationConfig

log = logging.getLogger(__name__)


def run_scrublet(adata: anndata.AnnData, config: CurationConfig) -> anndata.AnnData:
    """Detect and remove doublets with Scrublet."""
    if not config.remove_doublets:
        log.info("Doublet removal disabled — skipping Scrublet")
        return adata

    import scrublet as scr

    log.info("Running Scrublet (n_obs=%d)", adata.n_obs)
    scrub = scr.Scrublet(adata.X, random_state=config.random_seed)
    doublet_scores, predicted_doublets = scrub.scrub_doublets(verbose=False)

    adata.obs["doublet_score"] = doublet_scores
    adata.obs["predicted_doublet"] = predicted_doublets

    n_doublets = int(np.sum(predicted_doublets))
    log.info(
        "Scrublet: %d predicted doublets (%.1f%%)",
        n_doublets, 100.0 * n_doublets / adata.n_obs,
    )

    adata = adata[~adata.obs["predicted_doublet"]].copy()
    log.info("After doublet removal: %d cells remain", adata.n_obs)
    return adata
