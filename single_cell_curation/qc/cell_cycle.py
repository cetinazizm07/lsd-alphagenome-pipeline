from __future__ import annotations

import logging

import anndata

from single_cell_curation.config import CurationConfig
from single_cell_curation.qc._cc_genes import S_GENES, G2M_GENES

log = logging.getLogger(__name__)


def run_cell_cycle(
    adata: anndata.AnnData,
    config: CurationConfig,
    s_genes: list[str] | None = None,
    g2m_genes: list[str] | None = None,
) -> anndata.AnnData:
    """Score cell cycle phase using scanpy's score_genes_cell_cycle.

    Uses the standard Tirosh et al. 2016 gene lists by default.
    Scoring is done on log-normalised values (layers["log_norm"] when present,
    otherwise adata.X is assumed to be log-normalised).
    """
    if not config.score_cell_cycle:
        log.info("Cell cycle scoring disabled — skipping")
        return adata

    import scanpy as sc

    s = s_genes or S_GENES
    g2m = g2m_genes or G2M_GENES

    # Use standard log-normalised layer to avoid issues with PFlogPF negatives
    adata_tmp = adata.copy()
    if "log_norm" in adata.layers:
        adata_tmp.X = adata_tmp.layers["log_norm"]

    sc.tl.score_genes_cell_cycle(adata_tmp, s_genes=s, g2m_genes=g2m)

    adata.obs["S_score"] = adata_tmp.obs["S_score"]
    adata.obs["G2M_score"] = adata_tmp.obs["G2M_score"]
    adata.obs["phase"] = adata_tmp.obs["phase"]

    log.info(
        "Cell cycle phases: %s",
        adata.obs["phase"].value_counts().to_dict(),
    )
    return adata
