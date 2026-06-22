from __future__ import annotations

import logging
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from single_cell_curation.config import CurationConfig

log = logging.getLogger(__name__)


def load_local(config: CurationConfig) -> anndata.AnnData:
    path = config.local_data_path
    if path is None:
        raise ValueError("local_data_path must be set when source='local'")
    if config.local_format == "h5ad":
        return load_h5ad(path)
    elif config.local_format == "10x_mtx":
        return load_10x_mtx(path)
    elif config.local_format == "csv":
        return load_csv(path)
    raise ValueError(f"Unknown local_format: {config.local_format!r}")


def load_h5ad(path: Path) -> anndata.AnnData:
    log.info("Loading H5AD: %s", path)
    adata = anndata.read_h5ad(path)
    log.info("  %d cells × %d genes", adata.n_obs, adata.n_vars)
    return adata


def load_10x_mtx(path: Path) -> anndata.AnnData:
    import scanpy as sc
    log.info("Loading 10x MTX: %s", path)
    adata = sc.read_10x_mtx(path, var_names="gene_symbols", cache=False)
    adata.var_names_make_unique()
    log.info("  %d cells × %d genes", adata.n_obs, adata.n_vars)
    return adata


def load_csv(path: Path) -> anndata.AnnData:
    log.info("Loading CSV: %s", path)
    sep = "\t" if path.suffix in {".tsv", ".txt"} else ","
    compression = "gzip" if str(path).endswith(".gz") else None
    df = pd.read_csv(path, index_col=0, sep=sep, compression=compression)
    # If more rows than columns, assume genes × cells and transpose to cells × genes
    if df.shape[0] > df.shape[1]:
        df = df.T
    X = sp.csr_matrix(df.values.astype(np.float32))
    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame(index=df.index),
        var=pd.DataFrame(index=df.columns),
    )
    log.info("  %d cells × %d genes", adata.n_obs, adata.n_vars)
    return adata
