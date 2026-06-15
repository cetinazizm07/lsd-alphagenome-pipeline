"""
Real AlphaGenome ISM demo — SERPINE1 mesenchymal GBM reporter.

Gene       : SERPINE1 (PAI-1), chr7 hg38 — top mesenchymal GBM marker
             (Gargiulo et al. Nature Communications 2024, mes_coreg.regions.hg19.bed)
CRE window : 100 bp upstream of SERPINE1 TSS (chr7:100,546,778-100,546,878, hg38)
Context    : 1 Mbp window centred on TSS (AlphaGenome architecture requirement)
Scorers    : ATAC  — CenterMaskScorer, DIFF_LOG2_SUM, w=501
             RNA   — GeneMaskLFCScorer
Tracks used:
  ATAC — "motor neuron" (CL:0000100): closest available neural track
  RNA  — U-87 MG + A172 + M059J (all GBM cell lines in the model)
Fitness    : harmonic mean of |ATAC_log2FC| and |RNA_log2FC| (mean across GBM lines)

Output: demo/output/serpine1_ism_results.csv
        demo/output/serpine1_wt_cre.txt
        demo/output/top20_variants.tsv

Usage:
    ALPHAGENOME_API_KEY=... python demo/run_real_alphagenome.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alphagenome.models import dna_client, variant_scorers as vs_lib
from alphagenome.data import genome

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("ALPHAGENOME_API_KEY", "")
if not API_KEY:
    sys.exit("Set $ALPHAGENOME_API_KEY first.")

CHROM    = "chr7"
TSS_POS  = 100_546_878          # SERPINE1 TSS, hg38 (RefSeq NM_000602, + strand)

CTX_HALF       = dna_client.SEQUENCE_LENGTH_1MB // 2
INTERVAL_START = TSS_POS - CTX_HALF
INTERVAL_END   = INTERVAL_START + dna_client.SEQUENCE_LENGTH_1MB

# 100 bp CRE window immediately upstream of TSS
ISM_START = TSS_POS - 100
ISM_END   = TSS_POS             # half-open

# Cell type filters (exact biosample_name strings from AlphaGenome var metadata)
GBM_RNA_LINES   = {"U-87 MG", "A172", "M059J"}   # GBM cell lines in RNA-seq tracks
NEURAL_ATAC_KEY = "motor neuron"                   # only neural-type ATAC track available

EPS = 1e-9
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

SCORERS = [
    vs_lib.CenterMaskScorer(
        requested_output=dna_client.OutputType.ATAC,
        width=501,
        aggregation_type=vs_lib.AggregationType.DIFF_LOG2_SUM,
    ),
    vs_lib.GeneMaskLFCScorer(
        requested_output=dna_client.OutputType.RNA_SEQ,
    ),
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n{'='*68}")
    print("  SERPINE1 Mesenchymal GBM Reporter — AlphaGenome ISM (real API)")
    print(f"{'='*68}")
    print(f"  Gene     : SERPINE1 (PAI-1) — hg38 {CHROM}")
    print(f"  TSS      : {CHROM}:{TSS_POS:,}  (+ strand)")
    print(f"  Context  : {CHROM}:{INTERVAL_START:,}-{INTERVAL_END:,}  (1 Mbp)")
    print(f"  CRE/ISM  : {CHROM}:{ISM_START:,}-{ISM_END:,}  ({ISM_END-ISM_START} bp)")
    print(f"  ATAC     : motor neuron (best available neural proxy)")
    print(f"  RNA      : U-87 MG + A172 + M059J  (GBM cell lines)")
    print(f"  Variants : {(ISM_END-ISM_START)*3}  (100 bp × 3 alts = saturation)")
    print()

    model = dna_client.create(API_KEY)
    print("  Connected to AlphaGenome API.")

    interval     = genome.Interval(CHROM, INTERVAL_START, INTERVAL_END)
    ism_interval = genome.Interval(CHROM, ISM_START, ISM_END)

    print(f"  Running ISM — 10 × 10 bp batches (server-side inference) ...")
    ism_results = model.score_ism_variants(
        interval=interval,
        ism_interval=ism_interval,
        variant_scorers=SCORERS,
        organism=dna_client.Organism.HOMO_SAPIENS,
        progress_bar=True,
    )
    # Structure: list[list[AnnData]] — one [atac_adata, rna_adata] per variant
    print(f"  Done. {len(ism_results)} variants returned.\n")

    # -----------------------------------------------------------------------
    # Identify track indices once from the first element
    # -----------------------------------------------------------------------
    atac_var = ism_results[0][0].var
    rna_var  = ism_results[0][1].var

    # Motor neuron ATAC index
    atac_neural_idx = atac_var.index[
        atac_var["biosample_name"] == NEURAL_ATAC_KEY
    ].tolist()

    # GBM RNA indices
    rna_gbm_idx = rna_var.index[
        rna_var["biosample_name"].isin(GBM_RNA_LINES)
    ].tolist()
    rna_gbm_names = rna_var.loc[rna_gbm_idx, "biosample_name"].tolist()

    print(f"  ATAC track   : {NEURAL_ATAC_KEY}  (idx {atac_neural_idx})")
    print(f"  RNA tracks   : {rna_gbm_names}")
    print(f"  RNA n_tracks : {len(rna_gbm_idx)}")
    print()

    if not atac_neural_idx:
        print("  WARNING: motor neuron ATAC track not found; using all tracks.")
    if not rna_gbm_idx:
        print("  WARNING: no GBM RNA tracks found; using all tracks.")

    # -----------------------------------------------------------------------
    # Parse each variant
    # -----------------------------------------------------------------------
    rows = []
    for variant_result in ism_results:
        atac_adata = variant_result[0]
        rna_adata  = variant_result[1]

        v      = atac_adata.uns["variant"]  # genome.Variant
        pos_1b = v.position                 # 1-based VCF position
        ref    = v.reference_bases
        alt    = v.alternate_bases

        atac_x = np.asarray(atac_adata.X)[0]  # shape (n_atac_tracks,)
        rna_x  = np.asarray(rna_adata.X)[0]   # shape (n_rna_tracks,)

        # ATAC: neural proxy
        if atac_neural_idx:
            col = atac_var.index.get_loc(atac_neural_idx[0])
            atac_lfc = float(atac_x[col])
        else:
            atac_lfc = float(np.nanmean(atac_x))

        # RNA: mean across GBM lines
        if rna_gbm_idx:
            gbm_cols = [rna_var.index.get_loc(i) for i in rna_gbm_idx]
            rna_lfc  = float(np.nanmean(rna_x[gbm_cols]))
        else:
            rna_lfc  = float(np.nanmean(rna_x))

        # Individual GBM line scores
        gbm_scores = {}
        for idx, name in zip(rna_gbm_idx, rna_gbm_names):
            col = rna_var.index.get_loc(idx)
            gbm_scores[f"rna_{name.replace('-','').replace(' ','_')}"] = float(rna_x[col])

        # Harmonic mean fitness of absolute log2FC values
        a = abs(atac_lfc)
        r = abs(rna_lfc)
        fitness = (2 * a * r) / (a + r + EPS)

        row = {
            "chrom":        CHROM,
            "pos_0based":   pos_1b - 1,
            "pos_1based":   pos_1b,
            "ref":          ref,
            "alt":          alt,
            "atac_log2fc":  atac_lfc,       # motor neuron ATAC log2FC
            "rna_log2fc":   rna_lfc,        # mean GBM RNA log2FC
            "fitness":      fitness,
        }
        row.update(gbm_scores)
        rows.append(row)

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values("fitness", ascending=False).reset_index(drop=True)
    df_sorted.insert(0, "rank", df_sorted.index + 1)

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print(f"{'='*68}")
    print("  SERPINE1 CRE — WT vs In Silico Mutagenesis (AlphaGenome)")
    print(f"{'='*68}\n")

    print(f"  WT (reference) baseline:")
    print(f"    CRE coords  : {CHROM}:{ISM_START:,}-{ISM_END:,}  (hg38)")
    print(f"    ATAC log2FC : 0.0000  (motor neuron — reference state)")
    print(f"    RNA  log2FC : 0.0000  (GBM lines — reference state)")
    print(f"    Fitness     : 0.0000  (all mutants scored relative to WT)")
    print()

    print(f"  TOP 20 MUTANTS (ranked by harmonic-mean |log2FC| fitness)\n")
    print(f"  {'Rk':<4} {'Pos':>12} {'Var':>5}  "
          f"{'ATAC_l2fc':>10}  {'RNA_l2fc':>10}  {'Fitness':>8}  {'Dir':>4}")
    print("  " + "-" * 65)

    for _, row in df_sorted.head(20).iterrows():
        direction = "+" if row["atac_log2fc"] > 0 else "-"
        print(f"  {int(row['rank']):<4} "
              f"{CHROM}:{int(row['pos_1based']):>10}  "
              f"{row['ref']}>{row['alt']}  "
              f"{row['atac_log2fc']:>+10.4f}  "
              f"{row['rna_log2fc']:>+10.4f}  "
              f"{row['fitness']:>8.4f}  "
              f"{direction:>4}")

    print()

    # Summary
    gof = df[df["atac_log2fc"] > 0]
    lof = df[df["atac_log2fc"] < 0]
    best_row  = df_sorted.iloc[0]
    worst_row = df.sort_values("atac_log2fc").iloc[0]

    print(f"  Library summary ({len(df)} variants, 100 bp saturation):")
    print(f"    Gain-of-function (ATAC > 0) : {len(gof):>3}  variants")
    print(f"    Loss-of-function (ATAC < 0) : {len(lof):>3}  variants")
    print(f"    Mean ATAC log2FC            : {df['atac_log2fc'].mean():>+.4f}")
    print(f"    Mean RNA  log2FC            : {df['rna_log2fc'].mean():>+.4f}")
    print()
    print(f"  Best mutant (highest |fitness|):")
    print(f"    {CHROM}:{int(best_row['pos_1based']):,}  {best_row['ref']}>{best_row['alt']}"
          f"  ATAC={best_row['atac_log2fc']:+.4f}  RNA={best_row['rna_log2fc']:+.4f}"
          f"  fitness={best_row['fitness']:.4f}")
    print()
    print(f"  Most disruptive (lowest ATAC):")
    print(f"    {CHROM}:{int(worst_row['pos_1based']):,}  {worst_row['ref']}>{worst_row['alt']}"
          f"  ATAC={worst_row['atac_log2fc']:+.4f}  RNA={worst_row['rna_log2fc']:+.4f}")

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------
    csv_path = OUT_DIR / "serpine1_ism_results.csv"
    df_sorted.to_csv(csv_path, index=False)

    tsv_path = OUT_DIR / "top20_variants.tsv"
    df_sorted.head(20).to_csv(tsv_path, sep="\t", index=False)

    wt_path = OUT_DIR / "serpine1_wt_cre.txt"
    wt_path.write_text(
        f"SERPINE1 WT CRE — AlphaGenome ISM Analysis\n"
        f"Assembly    : hg38\n"
        f"Coordinates : {CHROM}:{ISM_START}-{ISM_END}  (0-based, half-open)\n"
        f"Length      : {ISM_END - ISM_START} bp\n"
        f"Gene        : SERPINE1 (PAI-1, NM_000602)  +strand\n"
        f"Cell type   : Glioblastoma mesenchymal (Gargiulo et al. 2024)\n"
        f"ATAC proxy  : motor neuron (CL:0000100) — closest neural track\n"
        f"RNA lines   : U-87 MG, A172, M059J (GBM cell lines)\n"
        f"Scorer ATAC : CenterMaskScorer DIFF_LOG2_SUM w=501\n"
        f"Scorer RNA  : GeneMaskLFCScorer\n"
        f"Fitness     : 2*|ATAC|*|RNA| / (|ATAC|+|RNA|+eps)\n"
        f"Source      : Gargiulo et al. Nature Communications 2024\n"
        f"             gitlab.com/gargiulo_lab/LSD_framework\n",
        encoding="utf-8",
    )

    print(f"\n  Exported:")
    print(f"    {csv_path}")
    print(f"    {tsv_path}")
    print(f"    {wt_path}")
    print()


if __name__ == "__main__":
    main()
