from __future__ import annotations

import logging
from pathlib import Path

import anndata

from single_cell_curation.config import CurationConfig
from single_cell_curation.ingest.local import load_h5ad, load_10x_mtx, load_csv
from single_cell_curation.utils.r_runner import RRunner

log = logging.getLogger(__name__)


def load_geo(config: CurationConfig) -> anndata.AnnData:
    """Download a GEO dataset via GEOquery R and load the result."""
    if not config.geo_accession:
        raise ValueError("geo_accession must be set when source='geo'")

    staging_dir = config.output_dir / "staging" / config.geo_accession
    staging_dir.mkdir(parents=True, exist_ok=True)

    runner = RRunner()
    runner.run(
        "geoquery.R",
        {"--accession": config.geo_accession, "--staging-dir": str(staging_dir)},
    )
    return _load_from_staging(staging_dir)


def _load_from_staging(staging_dir: Path) -> anndata.AnnData:
    """Auto-detect format and load from the GEO staging directory."""
    h5ad_files = sorted(staging_dir.rglob("*.h5ad"))
    if h5ad_files:
        return load_h5ad(h5ad_files[0])

    mtx_files = sorted(staging_dir.rglob("matrix.mtx*"))
    if mtx_files:
        return load_10x_mtx(mtx_files[0].parent)

    csv_candidates = [
        f for f in staging_dir.rglob("*")
        if f.suffix in {".csv", ".tsv", ".txt", ".gz"}
        and any(kw in f.name.lower() for kw in ("count", "matrix", "expression"))
    ]
    if csv_candidates:
        return load_csv(csv_candidates[0])

    raise FileNotFoundError(
        f"No loadable data file found in GEO staging dir {staging_dir}. "
        "Set source='local' and local_data_path to the downloaded file instead."
    )
