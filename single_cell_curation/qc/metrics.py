from __future__ import annotations

import logging

import anndata
import scanpy as sc

from single_cell_curation.config import CurationConfig

log = logging.getLogger(__name__)


def run_qc_metrics(adata: anndata.AnnData, config: CurationConfig) -> anndata.AnnData:
    """Compute QC metrics and apply cell / gene threshold filters."""
    log.info("QC metrics (n_obs=%d, n_vars=%d)", adata.n_obs, adata.n_vars)

    adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-"))
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    n_cells_before = adata.n_obs
    n_genes_before = adata.n_vars

    sc.pp.filter_genes(adata, min_cells=config.min_cells)
    sc.pp.filter_cells(adata, min_genes=config.min_genes)
    adata = adata[adata.obs["n_genes_by_counts"] <= config.max_genes].copy()
    adata = adata[adata.obs["pct_counts_mt"] <= config.max_pct_mito].copy()

    log.info(
        "After QC: %d→%d cells, %d→%d genes",
        n_cells_before, adata.n_obs,
        n_genes_before, adata.n_vars,
    )
    return adata
