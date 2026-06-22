from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import anndata
import numpy as np
import scipy.io
import scipy.sparse as sp

from single_cell_curation.config import CurationConfig

log = logging.getLogger(__name__)


def run_soupx(adata: anndata.AnnData, config: CurationConfig) -> anndata.AnnData:
    """Ambient RNA removal via SoupX (r_scripts/soupx.R).

    Requires config.cellranger_raw_dir pointing to the raw (unfiltered)
    CellRanger matrix directory. Skipped gracefully when Rscript or the
    raw dir is unavailable.
    """
    if not config.run_soupx:
        return adata

    if config.cellranger_raw_dir is None:
        log.warning(
            "SoupX skipped: config.cellranger_raw_dir is not set. "
            "Provide the CellRanger raw matrix directory to enable ambient correction."
        )
        return adata

    try:
        from single_cell_curation.utils.r_runner import RRunner
        runner = RRunner()
    except RuntimeError as exc:
        log.warning("SoupX skipped: %s", exc)
        return adata

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        filtered_dir = tmp / "filtered"
        output_dir = tmp / "corrected"
        _write_10x_mtx(adata, filtered_dir)

        runner.run(
            "soupx.R",
            {
                "--filtered": str(filtered_dir),
                "--raw": str(config.cellranger_raw_dir),
                "--output-dir": str(output_dir),
            },
        )
        return _read_corrected(adata, output_dir)


def _write_10x_mtx(adata: anndata.AnnData, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mat = adata.X.tocsc() if sp.issparse(adata.X) else sp.csc_matrix(adata.X)
    scipy.io.mmwrite(str(out_dir / "matrix.mtx"), mat)
    (out_dir / "barcodes.tsv").write_text("\n".join(adata.obs_names) + "\n")
    lines = [f"{g}\t{g}\tGene Expression" for g in adata.var_names]
    (out_dir / "features.tsv").write_text("\n".join(lines) + "\n")


def _read_corrected(adata: anndata.AnnData, corrected_dir: Path) -> anndata.AnnData:
    import scanpy as sc
    corrected = sc.read_10x_mtx(
        corrected_dir, var_names="gene_symbols", make_unique=False, cache=False
    )
    # Align to original adata (SoupX may reorder)
    common_genes = adata.var_names.intersection(corrected.var_names)
    adata_out = adata[:, common_genes].copy()
    adata_out.X = corrected[adata_out.obs_names, common_genes].X.astype(np.float32)
    log.info(
        "SoupX correction applied: %d cells × %d genes", adata_out.n_obs, adata_out.n_vars
    )
    return adata_out
