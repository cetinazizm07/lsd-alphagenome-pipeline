from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from single_cell_curation.config import CurationConfig

log = logging.getLogger(__name__)

# UCSC Xena TCGA hub — HTSeq read counts + GDC phenotype
_XENA_HUB = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download"
_DATASETS: dict[str, dict[str, str]] = {
    "TCGA-GBM": {
        "counts": f"{_XENA_HUB}/TCGA-GBM.htseq_counts.tsv.gz",
        "meta": f"{_XENA_HUB}/TCGA-GBM.GDC_phenotype.tsv.gz",
    },
    "TCGA-LGG": {
        "counts": f"{_XENA_HUB}/TCGA-LGG.htseq_counts.tsv.gz",
        "meta": f"{_XENA_HUB}/TCGA-LGG.GDC_phenotype.tsv.gz",
    },
    "TCGA-BRCA": {
        "counts": f"{_XENA_HUB}/TCGA-BRCA.htseq_counts.tsv.gz",
        "meta": f"{_XENA_HUB}/TCGA-BRCA.GDC_phenotype.tsv.gz",
    },
}


def load_tcga(config: CurationConfig) -> anndata.AnnData:
    project = config.tcga_project
    if project not in _DATASETS:
        raise ValueError(
            f"TCGA project {project!r} is not in the built-in map. "
            f"Known: {sorted(_DATASETS)}. "
            "Use source='xena' with xena_dataset_url for other projects."
        )
    staging_dir = config.output_dir / "staging" / project
    staging_dir.mkdir(parents=True, exist_ok=True)

    urls = _DATASETS[project]
    counts_path = _fetch_cached(urls["counts"], staging_dir)
    meta_path = _fetch_cached(urls["meta"], staging_dir)

    log.info("Loading TCGA counts from %s", counts_path.name)
    counts_df = pd.read_csv(counts_path, sep="\t", index_col=0, compression="gzip")
    log.info("Loading TCGA metadata from %s", meta_path.name)
    meta_df = pd.read_csv(meta_path, sep="\t", index_col=0, compression="gzip")

    return _build_adata(counts_df, meta_df)


def _fetch_cached(url: str, staging_dir: Path) -> Path:
    fname = url.split("/")[-1]
    dest = staging_dir / fname
    if dest.exists():
        log.info("Cached: %s", dest.name)
        return dest
    log.info("Downloading %s → %s", url, dest)
    urllib.request.urlretrieve(url, dest)
    return dest


def _build_adata(counts_df: pd.DataFrame, meta_df: pd.DataFrame) -> anndata.AnnData:
    # HTSeq counts: rows=genes, cols=samples — transpose to cells × genes
    shared = counts_df.columns.intersection(meta_df.index)
    counts_df = counts_df[shared]
    meta_df = meta_df.loc[shared]
    X = sp.csr_matrix(counts_df.T.values.astype(np.float32))
    adata = anndata.AnnData(
        X=X,
        obs=meta_df,
        var=pd.DataFrame(index=counts_df.index),
    )
    log.info("TCGA AnnData: %d samples × %d genes", adata.n_obs, adata.n_vars)
    return adata
