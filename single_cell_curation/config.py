from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

log = logging.getLogger(__name__)

Modality = Literal["single_cell", "bulk"]
Source = Literal["geo", "tcga", "xena", "cbioportal", "local"]
LocalFormat = Literal["10x_mtx", "h5ad", "csv"]
Assembly = Literal["hg38", "hg19", "mm10"]


@dataclass
class CurationConfig:
    """All parameters for a single-cell / bulk RNA-seq curation run."""

    dataset_name: str
    modality: Modality
    source: Source
    output_dir: Path

    # Source-specific
    geo_accession: str = ""
    tcga_project: str = "TCGA-GBM"
    xena_dataset_url: str = ""
    cbioportal_study_id: str = ""
    local_data_path: Path | None = None
    local_format: LocalFormat = "h5ad"

    # Genome
    assembly: Assembly = "hg38"

    # scRNA-seq QC thresholds
    min_genes: int = 200
    max_genes: int = 8_000
    min_cells: int = 3
    max_pct_mito: float = 20.0
    remove_doublets: bool = True
    doublet_score_threshold: float = 0.25
    run_soupx: bool = True
    # Path to CellRanger raw (unfiltered) matrix directory — required for SoupX
    cellranger_raw_dir: Path | None = None

    # Processing
    n_top_genes: int = 3_000
    leiden_resolution: float = 0.5
    n_pcs: int = 50
    n_neighbors: int = 15

    # Cell cycle
    score_cell_cycle: bool = True
    regress_cell_cycle: bool = False

    # Bulk-specific
    run_deseq2_vst: bool = False

    random_seed: int = 42

    def to_yaml(self, path: Path) -> None:
        data = {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in self.__dict__.items()
        }
        path.write_text(yaml.dump(data, sort_keys=True), encoding="utf-8")
        log.info("Config written to %s", path)

    @classmethod
    def from_yaml(cls, path: Path) -> "CurationConfig":
        raw: dict = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key in ("output_dir", "local_data_path", "cellranger_raw_dir"):
            if raw.get(key) is not None:
                raw[key] = Path(raw[key])
        return cls(**raw)
