from __future__ import annotations

import logging
from pathlib import Path

import anndata
import pandas as pd

from single_cell_curation.config import CurationConfig

log = logging.getLogger(__name__)


def write_outputs(adata: anndata.AnnData, config: CurationConfig) -> Path:
    """Write H5AD, counts CSV, and metadata TSV to config.output_dir."""
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)

    h5ad_path = out / f"{config.dataset_name}.h5ad"
    adata.write_h5ad(h5ad_path)
    log.info("H5AD written: %s", h5ad_path)

    # Raw-like counts matrix (cells × genes) — export log_norm layer when available
    if "log_norm" in adata.layers:
        counts_df = pd.DataFrame(
            adata.layers["log_norm"].toarray()
            if hasattr(adata.layers["log_norm"], "toarray")
            else adata.layers["log_norm"],
            index=adata.obs_names,
            columns=adata.var_names,
        )
    else:
        import scipy.sparse as sp
        X = adata.X
        counts_df = pd.DataFrame(
            X.toarray() if sp.issparse(X) else X,
            index=adata.obs_names,
            columns=adata.var_names,
        )
    counts_path = out / f"{config.dataset_name}_counts.csv"
    counts_df.T.to_csv(counts_path)  # genes × cells for downstream tools
    log.info("Counts CSV written: %s", counts_path)

    meta_path = out / f"{config.dataset_name}_metadata.tsv"
    adata.obs.to_csv(meta_path, sep="\t")
    log.info("Metadata TSV written: %s", meta_path)

    return h5ad_path
