"""
CLI entry point.

lsd-alpha run --genes input/sig.bed \
              --tf input/tf.txt \
              --cell "Glioblastoma" \
              --strategy "saturation" \
              --out "./output" \
              --assembly hg38 \
              --fasta reference/genome.fa \
              --api-key $ALPHAGENOME_API_KEY

All flags map 1:1 to PipelineConfig fields.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click

from lsd_alphagenome_pipeline.config import PipelineConfig, resolve_cell_ontology
from lsd_alphagenome_pipeline.pipeline import Pipeline


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
    """LSD + AlphaGenome super-promoter pipeline."""


@main.command("run")
@click.option("--genes", "genes_bed", required=True, type=click.Path(exists=True, path_type=Path),
              help="6-column BED file of signature genes.")
@click.option("--tf", "tf_motifs", required=True, type=click.Path(exists=True, path_type=Path),
              help="MEME-format motif file (JASPAR / HOCOMOCO).")
@click.option("--cell", "cell_type", required=True,
              help="Cell type name (e.g. 'Glioblastoma') or CURIE (e.g. 'CL:0001063').")
@click.option("--strategy", "strategy",
              type=click.Choice(["saturation", "spacer"], case_sensitive=False),
              default="saturation", show_default=True)
@click.option("--out", "output_dir", required=True, type=click.Path(path_type=Path),
              help="Output directory (created if absent).")
@click.option("--assembly", default="hg38", show_default=True,
              type=click.Choice(["hg38", "hg19", "mm10", "mm39"]),
              help="Reference genome assembly. Must match your BED coordinates.")
@click.option("--fasta", "fasta", default="reference/genome.fa",
              type=click.Path(path_type=Path),
              help="Path to reference FASTA (must be indexed).")
@click.option("--ctcf-bed", "ctcf_bed", default=None,
              type=click.Path(exists=True, path_type=Path),
              help="BED of CTCF sites for locus definition (optional).")
@click.option("--api-key", "api_key",
              default=lambda: os.environ.get("ALPHAGENOME_API_KEY", ""),
              help="AlphaGenome API key. Defaults to $ALPHAGENOME_API_KEY.")
@click.option("--max-variants", "max_variants", default=5000, show_default=True, type=int,
              help="Hard cap on number of variants queried per run.")
@click.option("--seed", "random_seed", default=42, show_default=True, type=int,
              help="Random seed for spacer shuffling reproducibility.")
@click.option("--top-n", "top_n_export", default=20, show_default=True, type=int,
              help="Number of top-ranked variants to export.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Enable DEBUG-level logging.")
def run(
    genes_bed: Path,
    tf_motifs: Path,
    cell_type: str,
    strategy: str,
    output_dir: Path,
    assembly: str,
    fasta: Path,
    ctcf_bed: Path | None,
    api_key: str,
    max_variants: int,
    random_seed: int,
    top_n_export: int,
    verbose: bool,
) -> None:
    """Run the full LSD → mutagenesis → AlphaGenome → ranking pipeline."""
    _configure_logging(verbose)
    log = logging.getLogger(__name__)

    # Validate cell type early so we fail before any heavy computation
    try:
        ontology_id = resolve_cell_ontology(cell_type)
        log.info("Cell type '%s' resolved to ontology ID: %s", cell_type, ontology_id)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if not api_key:
        click.echo(
            "Error: AlphaGenome API key is required. "
            "Pass --api-key or set $ALPHAGENOME_API_KEY.",
            err=True,
        )
        sys.exit(1)

    config = PipelineConfig(
        genes_bed=genes_bed,
        tf_motifs=tf_motifs,
        output_dir=output_dir,
        assembly=assembly,          # type: ignore[arg-type]
        fasta=fasta,
        ctcf_bed=ctcf_bed,
        strategy=strategy,          # type: ignore[arg-type]
        max_variants=max_variants,
        random_seed=random_seed,
        alphagenome_api_key=api_key,
        cell_type=cell_type,
        top_n_export=top_n_export,
    )

    log.info("Starting pipeline run → %s", output_dir)
    pipeline = Pipeline(config)
    try:
        csv_path = pipeline.run()
        click.echo(f"Done. Results: {csv_path}")
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(2)
    except RuntimeError as exc:
        click.echo(f"Pipeline error: {exc}", err=True)
        sys.exit(3)


@main.command("validate-config")
@click.argument("config_yaml", type=click.Path(exists=True, path_type=Path))
def validate_config(config_yaml: Path) -> None:
    """Load and validate a saved YAML config without running the pipeline."""
    try:
        config = PipelineConfig.from_yaml(config_yaml)
        click.echo(f"Config valid. Cell ontology: {config.cell_ontology_id()}")
        click.echo(f"Assembly: {config.assembly}")
        click.echo(f"Strategy: {config.strategy}")
    except Exception as exc:
        click.echo(f"Config error: {exc}", err=True)
        sys.exit(1)
