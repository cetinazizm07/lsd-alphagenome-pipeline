"""
sc-curate CLI

sc-curate run-sc  --config run.yaml
sc-curate run-sc  --name Darmanis --source local --input data.h5ad --out results/
sc-curate run-bulk --config run.yaml
sc-curate download --source geo --accession GSE84465 --out staging/
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from single_cell_curation.config import CurationConfig
from single_cell_curation.pipeline import CurationPipeline


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
    )


@click.group()
def main() -> None:
    """Single-cell curation pipeline — normalize, QC, cluster, export."""


# ---------------------------------------------------------------------------
# run-sc
# ---------------------------------------------------------------------------

@main.command("run-sc")
@click.option("--config", "config_yaml", default=None,
              type=click.Path(exists=True, path_type=Path),
              help="YAML config file. All other flags are ignored when provided.")
@click.option("--name", "dataset_name", default=None, help="Dataset name.")
@click.option("--source", default=None,
              type=click.Choice(["local", "geo", "tcga", "xena", "cbioportal"]),
              help="Data source.")
@click.option("--input", "local_data_path", default=None,
              type=click.Path(path_type=Path),
              help="Local data file (H5AD / 10x dir / CSV).")
@click.option("--format", "local_format", default="h5ad",
              type=click.Choice(["h5ad", "10x_mtx", "csv"]))
@click.option("--accession", "geo_accession", default="",
              help="GEO accession (e.g. GSE84465).")
@click.option("--out", "output_dir", required=True, type=click.Path(path_type=Path),
              help="Output directory.")
@click.option("--assembly", default="hg38",
              type=click.Choice(["hg38", "hg19", "mm10"]))
@click.option("--seed", "random_seed", default=42, show_default=True, type=int)
@click.option("--verbose", "-v", is_flag=True, default=False)
def run_sc(
    config_yaml: Path | None,
    dataset_name: str | None,
    source: str | None,
    local_data_path: Path | None,
    local_format: str,
    geo_accession: str,
    output_dir: Path,
    assembly: str,
    random_seed: int,
    verbose: bool,
) -> None:
    """Run the full single-cell curation pipeline."""
    _configure_logging(verbose)

    if config_yaml is not None:
        config = CurationConfig.from_yaml(config_yaml)
    else:
        if not dataset_name or not source:
            raise click.UsageError("--name and --source are required when not using --config.")
        config = CurationConfig(
            dataset_name=dataset_name,
            modality="single_cell",
            source=source,  # type: ignore[arg-type]
            output_dir=output_dir,
            local_data_path=local_data_path,
            local_format=local_format,  # type: ignore[arg-type]
            geo_accession=geo_accession,
            assembly=assembly,  # type: ignore[arg-type]
            random_seed=random_seed,
        )
        config.output_dir = output_dir

    try:
        result = CurationPipeline(config).run()
        click.echo(f"Done. Output: {result.output_dir}")
        click.echo(f"  {result.adata.n_obs} cells × {result.adata.n_vars} genes")
        click.echo(f"  Report: {result.output_dir / 'qc_report.json'}")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# run-bulk
# ---------------------------------------------------------------------------

@main.command("run-bulk")
@click.option("--config", "config_yaml", default=None,
              type=click.Path(exists=True, path_type=Path))
@click.option("--name", "dataset_name", default=None)
@click.option("--source", default=None,
              type=click.Choice(["local", "geo", "tcga", "xena", "cbioportal"]))
@click.option("--input", "local_data_path", default=None,
              type=click.Path(path_type=Path))
@click.option("--out", "output_dir", required=True, type=click.Path(path_type=Path))
@click.option("--vst", "run_deseq2_vst", is_flag=True, default=False,
              help="Apply DESeq2 VST normalization (requires R + DESeq2).")
@click.option("--verbose", "-v", is_flag=True, default=False)
def run_bulk(
    config_yaml: Path | None,
    dataset_name: str | None,
    source: str | None,
    local_data_path: Path | None,
    output_dir: Path,
    run_deseq2_vst: bool,
    verbose: bool,
) -> None:
    """Run the bulk RNA-seq curation pipeline."""
    _configure_logging(verbose)

    if config_yaml is not None:
        config = CurationConfig.from_yaml(config_yaml)
    else:
        if not dataset_name or not source:
            raise click.UsageError("--name and --source are required when not using --config.")
        config = CurationConfig(
            dataset_name=dataset_name,
            modality="bulk",
            source=source,  # type: ignore[arg-type]
            output_dir=output_dir,
            local_data_path=local_data_path,
            run_deseq2_vst=run_deseq2_vst,
        )

    try:
        result = CurationPipeline(config).run()
        click.echo(f"Done. Output: {result.output_dir}")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

@main.command("download")
@click.option("--source", required=True,
              type=click.Choice(["geo", "tcga", "xena", "cbioportal"]))
@click.option("--accession", default="", help="GEO accession or TCGA project.")
@click.option("--url", "xena_url", default="", help="Xena dataset URL.")
@click.option("--study", "cbioportal_study", default="", help="cBioPortal study ID.")
@click.option("--out", "output_dir", required=True, type=click.Path(path_type=Path))
@click.option("--verbose", "-v", is_flag=True, default=False)
def download(
    source: str,
    accession: str,
    xena_url: str,
    cbioportal_study: str,
    output_dir: Path,
    verbose: bool,
) -> None:
    """Download a dataset without running the full pipeline."""
    _configure_logging(verbose)
    config = CurationConfig(
        dataset_name=accession or cbioportal_study or "dataset",
        modality="single_cell",
        source=source,  # type: ignore[arg-type]
        output_dir=output_dir,
        geo_accession=accession,
        tcga_project=accession,
        xena_dataset_url=xena_url,
        cbioportal_study_id=cbioportal_study,
    )
    try:
        from single_cell_curation.ingest import load_dataset
        adata = load_dataset(config)
        click.echo(f"Downloaded: {adata.n_obs} cells × {adata.n_vars} genes")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# validate-config
# ---------------------------------------------------------------------------

@main.command("validate-config")
@click.argument("config_yaml", type=click.Path(exists=True, path_type=Path))
def validate_config(config_yaml: Path) -> None:
    """Validate a YAML config file without running the pipeline."""
    try:
        config = CurationConfig.from_yaml(config_yaml)
        click.echo(f"Config valid.")
        click.echo(f"  dataset_name : {config.dataset_name}")
        click.echo(f"  modality     : {config.modality}")
        click.echo(f"  source       : {config.source}")
        click.echo(f"  output_dir   : {config.output_dir}")
    except Exception as exc:
        click.echo(f"Config error: {exc}", err=True)
        sys.exit(1)
