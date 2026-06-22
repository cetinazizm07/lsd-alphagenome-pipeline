from __future__ import annotations

import logging
import tarfile
import urllib.request
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from single_cell_curation.config import CurationConfig

log = logging.getLogger(__name__)


def load_xena(config: CurationConfig) -> anndata.AnnData:
    if not config.xena_dataset_url:
        raise ValueError("xena_dataset_url must be set when source='xena'")
    staging_dir = config.output_dir / "staging" / "xena"
    staging_dir.mkdir(parents=True, exist_ok=True)
    data_path = _fetch_cached(config.xena_dataset_url, staging_dir)
    return _load_matrix_file(data_path)


def load_cbioportal(config: CurationConfig) -> anndata.AnnData:
    if not config.cbioportal_study_id:
        raise ValueError("cbioportal_study_id must be set when source='cbioportal'")
    study = config.cbioportal_study_id
    staging_dir = config.output_dir / "staging" / study
    staging_dir.mkdir(parents=True, exist_ok=True)

    archive_url = f"https://cbioportal-datahub.s3.amazonaws.com/{study}.tar.gz"
    archive_path = _fetch_cached(archive_url, staging_dir)
    _extract_tar_if_needed(archive_path, staging_dir)
    return _find_and_load_expression(staging_dir)


def _fetch_cached(url: str, staging_dir: Path) -> Path:
    fname = url.split("/")[-1].split("?")[0]
    dest = staging_dir / fname
    if dest.exists():
        log.info("Cached: %s", dest.name)
        return dest
    log.info("Downloading %s → %s", url, dest)
    urllib.request.urlretrieve(url, dest)
    return dest


def _extract_tar_if_needed(archive_path: Path, dest_dir: Path) -> None:
    marker = dest_dir / ".extracted"
    if marker.exists():
        return
    log.info("Extracting %s", archive_path.name)
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(dest_dir)
    marker.touch()


def _load_matrix_file(path: Path) -> anndata.AnnData:
    sep = "\t" if path.suffix in {".tsv", ".txt"} else ","
    compression = "gzip" if str(path).endswith(".gz") else None
    df = pd.read_csv(path, sep=sep, index_col=0, compression=compression)
    if df.shape[0] > df.shape[1]:
        df = df.T
    X = sp.csr_matrix(df.values.astype(np.float32))
    return anndata.AnnData(
        X=X,
        obs=pd.DataFrame(index=df.index),
        var=pd.DataFrame(index=df.columns),
    )


def _find_and_load_expression(study_dir: Path) -> anndata.AnnData:
    candidates = [
        f for f in study_dir.rglob("*")
        if f.suffix in {".txt", ".tsv"}
        and any(kw in f.name.lower() for kw in ("mrna", "expression", "rna_seq"))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find mRNA expression file in {study_dir}. "
            "Set source='xena' with xena_dataset_url for direct file access."
        )
    return _load_matrix_file(candidates[0])
