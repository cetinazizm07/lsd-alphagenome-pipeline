"""
Centralised run configuration.

All pipeline parameters live here so every module receives a single typed
object rather than scattered keyword arguments.  Serialising this to YAML
satisfies the reproducibility requirement: the same config file re-runs
identically on any machine with the same reference genome.

Ref: LSD paper (Nature Communications 2024) §Methods — parameter table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Genome assembly → UCSC-style chromosome name convention
# ---------------------------------------------------------------------------
Assembly = Literal["hg38", "hg19", "mm10", "mm39"]

# ---------------------------------------------------------------------------
# Ontology ID map: common cell-type names → UBERON / CL identifiers
# Source: https://www.ebi.ac.uk/ols4/ontologies/uberon
#         https://www.ebi.ac.uk/ols4/ontologies/cl
# AlphaGenome requires these exact CURIEs in ontology_terms.
# ---------------------------------------------------------------------------
CELL_TYPE_ONTOLOGY: dict[str, str] = {
    "glioblastoma": "CL:0001063",       # neoplastic cell of glioblastoma
    "glioblastoma stem cell": "CL:0001064",
    "neuron": "CL:0000540",
    "astrocyte": "CL:0000127",
    "microglia": "CL:0000129",
    "oligodendrocyte": "CL:0000128",
    "neural stem cell": "CL:0000047",
    "hela": "CL:0002621",               # HeLa cervical cancer cell
    "k562": "CL:0000037",               # K562 myelogenous leukemia
    "gm12878": "CL:0000236",            # B lymphocyte
    "hepg2": "CL:0001060",             # hepatocellular carcinoma cell
    "hek293": "CL:0002293",             # embryonic kidney cell
    "mcf7": "CL:0001063",               # breast cancer cell (reused CL for neoplastic)
    "imr90": "CL:0002328",              # lung fibroblast
}


def resolve_cell_ontology(cell_name: str) -> str:
    """
    Return ontology CURIE for *cell_name*.

    Accepts either a plain English name (looked up in CELL_TYPE_ONTOLOGY)
    or a pre-formed CURIE such as ``CL:0000540`` / ``UBERON:0000955``.

    Raises ValueError for unrecognised names so the pipeline fails early
    rather than silently querying AlphaGenome with an invalid term.
    """
    if ":" in cell_name:
        return cell_name  # caller supplied a CURIE directly
    key = cell_name.lower().strip()
    if key not in CELL_TYPE_ONTOLOGY:
        raise ValueError(
            f"Unknown cell type '{cell_name}'. "
            f"Pass a CURIE directly (e.g. 'CL:0000540') or add the mapping "
            f"to config.CELL_TYPE_ONTOLOGY. "
            f"Known names: {sorted(CELL_TYPE_ONTOLOGY)}"
        )
    return CELL_TYPE_ONTOLOGY[key]


# ---------------------------------------------------------------------------
# Main configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    All tunable parameters for a pipeline run.

    Reproducibility contract
    ------------------------
    * ``assembly`` pins the reference genome version; BED coordinates must
      match this assembly.
    * ``motif_pvalue_threshold`` and ``motif_bg_order`` replicate FIMO's
      p-value and background model settings (now implemented via MOODS).
    * ``random_seed`` controls spacer-shuffling order.
    * ``alphagenome_context_bp`` must be 1_048_576 (2^20) — AlphaGenome's
      required input window (Ref: AlphaGenome README §Interval size).
    """

    # Input files
    genes_bed: Path                     # 6-col BED of signature genes
    tf_motifs: Path                     # MEME-format motif file (JASPAR / HOCOMOCO)
    output_dir: Path

    # Genome
    assembly: Assembly = "hg38"
    fasta: Path = Path("reference/genome.fa")   # local reference FASTA

    # LSD selection
    # Ref: LSD paper §Methods "Locus definition" — CTCF BED is optional;
    # if absent the pipeline uses ±500 kb from gene TSS as boundary.
    ctcf_bed: Path | None = None
    locus_flank_bp: int = 500_000       # fallback boundary when no CTCF BED
    enrichment_top_n: int = 1           # number of CREs to return from LSD

    # Motif scanning (MOODS, replacing FIMO)
    # Ref: MOODS docs §Threshold — threshold = -log10(p_value) in bits
    motif_pvalue_threshold: float = 1e-4
    motif_bg_order: int = 1             # Markov order for background model

    # Mutagenesis
    strategy: Literal["saturation", "spacer"] = "saturation"
    max_variants: int = 5_000           # hard cap on API calls
    strategy_window: int | None = None  # bp, restrict saturation to sub-region
    spacer_max_delta_bp: int = 5        # ±N bp per spacer interval
    random_seed: int = 42

    # AlphaGenome
    alphagenome_api_key: str = ""
    cell_type: str = "glioblastoma"     # resolved to ontology CURIE at runtime
    alphagenome_context_bp: int = 1_048_576  # 2^20, fixed by model architecture
    alphagenome_max_retries: int = 5
    alphagenome_backoff_base: float = 2.0   # seconds, exponential base
    alphagenome_batch_size: int = 50        # variants per API session

    # Aggregation region for track → scalar
    # "cre": mean signal over the exact CRE genomic span
    # "tss_window": mean over ±N bp from the nearest annotated TSS
    track_agg_region: Literal["cre", "tss_window"] = "cre"
    tss_window_bp: int = 500            # used only when track_agg_region == "tss_window"

    # Fitness scoring
    fitness_epsilon: float = 1e-9       # ε in harmonic mean, prevents /0

    # Output
    top_n_export: int = 20              # number of ranked mutants to export

    def cell_ontology_id(self) -> str:
        """Resolve cell_type string → ontology CURIE."""
        return resolve_cell_ontology(self.cell_type)

    def fingerprint(self) -> str:
        """Return a deterministic string representation of parameters affecting computation."""
        keys = [
            "genes_bed", "tf_motifs", "assembly", "fasta", "ctcf_bed",
            "locus_flank_bp", "strategy", "max_variants", "strategy_window",
            "spacer_max_delta_bp", "random_seed", "cell_type",
            "motif_pvalue_threshold", "motif_bg_order", "track_agg_region"
        ]
        items = []
        for k in keys:
            v = getattr(self, k, None)
            items.append(f"{k}={str(v)}")
        return ";".join(items)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_yaml(self, path: Path, mask_key: bool = True) -> None:
        """Persist config to YAML for run reproducibility (masks API key by default)."""
        data = {}
        for k, v in self.__dict__.items():
            if k == "alphagenome_api_key" and mask_key and v:
                data[k] = "***MASKED***"
            else:
                data[k] = str(v) if isinstance(v, Path) else v
        path.write_text(yaml.dump(data, sort_keys=True), encoding="utf-8")
        log.info("Config written to %s", path)

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        raw: dict = yaml.safe_load(path.read_text(encoding="utf-8"))
        # Coerce Path fields
        for key in ("genes_bed", "tf_motifs", "output_dir", "fasta", "ctcf_bed"):
            if raw.get(key) is not None:
                raw[key] = Path(raw[key])
        return cls(**raw)
