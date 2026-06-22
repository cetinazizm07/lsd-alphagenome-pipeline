from __future__ import annotations

import logging

import anndata
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from single_cell_curation.config import CurationConfig

log = logging.getLogger(__name__)


def pflogpf_normalize(
    adata: anndata.AnnData,
    target_sum: float | None = None,
) -> anndata.AnnData:
    """PFlogPF normalization (Booeshaghi et al. 2022, doi:10.1101/2022.05.06.490859).

    Row PF → log1p → Column PF (gene mean subtraction).
    Equivalent to a shifted centered log-ratio (sCLR).

    The pre-column-PF log-normalised matrix is stored in layers["log_norm"]
    so tools requiring non-negative values (HVG selection, cell cycle scoring)
    can access it.
    """
    # Step 1 – Row PF: normalize each cell to the median library size
    lib_sizes = np.asarray(adata.X.sum(axis=1)).ravel()
    med_lib = float(target_sum or np.median(lib_sizes))
    sc.pp.normalize_total(adata, target_sum=med_lib)

    # Step 2 – log1p
    sc.pp.log1p(adata)
    adata.layers["log_norm"] = adata.X.copy()

    # Step 3 – Column PF: subtract per-gene mean (centers each gene)
    col_means = np.asarray(adata.X.mean(axis=0)).ravel()
    X_dense = adata.X.toarray() if sp.issparse(adata.X) else adata.X.copy()
    X_dense -= col_means
    adata.X = X_dense.astype(np.float32)

    log.info("PFlogPF normalization complete")
    return adata


def run_sc_pipeline(adata: anndata.AnnData, config: CurationConfig) -> anndata.AnnData:
    """PFlogPF → HVG → PCA → neighbors → Leiden → UMAP."""
    adata = pflogpf_normalize(adata)

    # HVG selection on log-normalised layer (non-negative)
    sc.pp.highly_variable_genes(
        adata, n_top_genes=config.n_top_genes, layer="log_norm", inplace=True
    )
    log.info("HVG: %d genes selected", int(adata.var["highly_variable"].sum()))

    # PCA on PFlogPF-normalised HVG subset (already centred — no additional scaling)
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.tl.pca(adata_hvg, n_comps=config.n_pcs, random_state=config.random_seed)
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    adata.uns["pca"] = adata_hvg.uns.get("pca", {})

    # Neighborhood graph, clustering, embedding
    sc.pp.neighbors(
        adata,
        n_neighbors=config.n_neighbors,
        n_pcs=config.n_pcs,
        random_state=config.random_seed,
    )
    sc.tl.leiden(adata, resolution=config.leiden_resolution, random_state=config.random_seed)
    sc.tl.umap(adata, random_state=config.random_seed)

    log.info(
        "sc pipeline done: %d cells, %d Leiden clusters",
        adata.n_obs, adata.obs["leiden"].nunique(),
    )
    return adata
