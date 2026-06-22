from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from single_cell_curation.config import CurationConfig

log = logging.getLogger(__name__)


def tpm_normalize(
    counts_df: pd.DataFrame,
    gene_lengths: pd.Series,
) -> pd.DataFrame:
    """Transcripts Per Million normalization.

    counts_df : genes × samples (raw integer counts)
    gene_lengths : gene lengths in base pairs, indexed by gene name
    """
    shared = counts_df.index.intersection(gene_lengths.index)
    counts_df = counts_df.loc[shared]
    lengths = gene_lengths.loc[shared].values[:, None]
    rpk = counts_df.values / (lengths / 1_000.0)
    tpm = rpk / rpk.sum(axis=0, keepdims=True) * 1_000_000.0
    return pd.DataFrame(tpm, index=counts_df.index, columns=counts_df.columns)


def run_bulk_pipeline(
    counts_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    config: CurationConfig,
    gene_lengths: pd.Series | None = None,
) -> anndata.AnnData:
    """Normalize bulk RNA-seq counts and optionally apply DESeq2 VST.

    counts_df  : genes × samples
    meta_df    : samples × metadata
    gene_lengths : optional; if provided, TPM normalization is applied before VST
    """
    if gene_lengths is not None:
        log.info("Applying TPM normalization")
        norm_df = tpm_normalize(counts_df, gene_lengths)
    else:
        log.info("No gene lengths provided — using raw counts")
        norm_df = counts_df.copy()

    if config.run_deseq2_vst:
        norm_df = _deseq2_vst(counts_df, config)  # VST uses raw counts

    # Build AnnData (cells = samples, vars = genes)
    shared_samples = norm_df.columns.intersection(meta_df.index)
    X = sp.csr_matrix(norm_df[shared_samples].T.values.astype(np.float32))
    adata = anndata.AnnData(
        X=X,
        obs=meta_df.loc[shared_samples],
        var=pd.DataFrame(index=norm_df.index),
    )
    log.info("Bulk AnnData: %d samples × %d genes", adata.n_obs, adata.n_vars)
    return adata


def _deseq2_vst(counts_df: pd.DataFrame, config: CurationConfig) -> pd.DataFrame:
    from single_cell_curation.utils.r_runner import RRunner
    runner = RRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        counts_path = tmp / "counts.tsv"
        output_path = tmp / "vst.tsv"

        counts_df.to_csv(counts_path, sep="\t")
        runner.run(
            "deseq2_vst.R",
            {"--counts-tsv": str(counts_path), "--output-tsv": str(output_path)},
        )
        vst_df = pd.read_csv(output_path, sep="\t", index_col=0)
        log.info("DESeq2 VST applied: %d genes × %d samples", *vst_df.shape)
        return vst_df
