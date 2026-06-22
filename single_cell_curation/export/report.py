from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import anndata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from single_cell_curation.config import CurationConfig

log = logging.getLogger(__name__)


def write_report(
    adata: anndata.AnnData,
    config: CurationConfig,
    extra_stats: dict[str, Any] | None = None,
) -> Path:
    """Write qc_report.json and QC/embedding PNGs to config.output_dir."""
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)

    stats = _collect_stats(adata, config)
    if extra_stats:
        stats.update(extra_stats)

    report_path = out / "qc_report.json"
    report_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    log.info("QC report written: %s", report_path)

    _write_qc_plots(adata, out)
    if config.modality == "single_cell" and "X_umap" in adata.obsm:
        _write_umap_plot(adata, out)

    return report_path


def _collect_stats(adata: anndata.AnnData, config: CurationConfig) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "dataset_name": config.dataset_name,
        "modality": config.modality,
        "source": config.source,
        "assembly": config.assembly,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
    }
    if "n_genes_by_counts" in adata.obs.columns:
        stats["median_genes_per_cell"] = float(adata.obs["n_genes_by_counts"].median())
    if "pct_counts_mt" in adata.obs.columns:
        stats["median_pct_mito"] = float(adata.obs["pct_counts_mt"].median())
    if "doublet_score" in adata.obs.columns:
        stats["n_doublets_removed"] = int(adata.obs.get("predicted_doublet", False).sum())
    if "leiden" in adata.obs.columns:
        stats["n_leiden_clusters"] = int(adata.obs["leiden"].nunique())
    if "phase" in adata.obs.columns:
        stats["cell_cycle_distribution"] = adata.obs["phase"].value_counts().to_dict()
    return stats


def _write_qc_plots(adata: anndata.AnnData, out: Path) -> None:
    qc_cols = [c for c in ("n_genes_by_counts", "total_counts", "pct_counts_mt")
               if c in adata.obs.columns]
    if not qc_cols:
        return
    fig, axes = plt.subplots(1, len(qc_cols), figsize=(5 * len(qc_cols), 4))
    if len(qc_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, qc_cols):
        ax.violinplot(adata.obs[col].values, showmedians=True)
        ax.set_title(col.replace("_", " "))
        ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(out / "qc_violin.png", dpi=120)
    plt.close(fig)
    log.info("QC violin plot written")


def _write_umap_plot(adata: anndata.AnnData, out: Path) -> None:
    import numpy as np
    coords = adata.obsm["X_umap"]
    color_col = "leiden" if "leiden" in adata.obs.columns else None

    fig, ax = plt.subplots(figsize=(6, 5))
    if color_col:
        categories = adata.obs[color_col].astype("category")
        palette = plt.cm.get_cmap("tab20", len(categories.cat.categories))
        for i, cat in enumerate(categories.cat.categories):
            mask = categories == cat
            ax.scatter(coords[mask, 0], coords[mask, 1], s=3, alpha=0.6,
                       color=palette(i), label=cat)
        ax.legend(markerscale=4, bbox_to_anchor=(1.02, 1), loc="upper left",
                  fontsize=7, title="Leiden")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=3, alpha=0.4, color="steelblue")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP")
    fig.tight_layout()
    fig.savefig(out / "umap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("UMAP plot written")
