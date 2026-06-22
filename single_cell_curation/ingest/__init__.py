from __future__ import annotations

import anndata

from single_cell_curation.config import CurationConfig
from single_cell_curation.ingest.local import load_local, load_h5ad, load_10x_mtx, load_csv
from single_cell_curation.ingest.geo import load_geo
from single_cell_curation.ingest.tcga import load_tcga
from single_cell_curation.ingest.xena_cbioportal import load_xena, load_cbioportal

__all__ = [
    "load_dataset",
    "load_local", "load_h5ad", "load_10x_mtx", "load_csv",
    "load_geo",
    "load_tcga",
    "load_xena", "load_cbioportal",
]

_DISPATCH = {
    "local": load_local,
    "geo": load_geo,
    "tcga": load_tcga,
    "xena": load_xena,
    "cbioportal": load_cbioportal,
}


def load_dataset(config: CurationConfig) -> anndata.AnnData:
    """Dispatch to the appropriate ingestor based on config.source."""
    loader = _DISPATCH.get(config.source)
    if loader is None:
        raise ValueError(f"Unknown source: {config.source!r}. Choose from: {sorted(_DISPATCH)}")
    return loader(config)
