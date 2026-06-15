# LSD × AlphaGenome Pipeline

Python pipeline integrating the **LSD framework** (Gargiulo et al., *Nature Communications* 2024) with the **AlphaGenome API** (Google DeepMind) to:

1. Select optimal Wild-Type CREs using the LSD scoring framework
2. Generate in silico mutagenesis variants with saturation scanning
3. Score all variants with AlphaGenome ISM and rank by a harmonic-mean fitness function

## Biological context

Demonstrated on **SERPINE1 (PAI-1)** in **glioblastoma mesenchymal** cells:
- CRE window: `chr7:100,546,778–100,546,878` (hg38), 100 bp upstream of TSS
- Cell type: mesenchymal GBM (Gargiulo et al. 2024 co-regulatory network)
- ATAC proxy: motor neuron (closest neural track in AlphaGenome)
- RNA tracks: U-87 MG, A172, M059J (GBM cell lines)
- Fitness: `2×|ATAC|×|RNA| / (|ATAC|+|RNA|+ε)` (harmonic mean of log2FC scores)

## Pipeline steps

```
Step 1 — LSD locus definition    CTCF-bounded genomic window around gene TSS
Step 2 — TFBS assessment         MOODS motif scanning with JASPAR PWMs
Step 3 — CRE selection           Enrichment-matrix scoring → best 150 bp window
Step 4 — Mutagenesis design      Saturation SNV library (3 alts × every position)
Step 5 — AlphaGenome ISM         score_ism_variants() → ATAC + RNA log2FC per variant
Step 6 — Ranking                 Harmonic-mean fitness, export CSV + report
```

## Installation

```bash
pip install -e .
```

**Requires:** Python ≥ 3.11, biopython, MOODS-python, alphagenome, tenacity, click, pybedtools

## Usage

```bash
export ALPHAGENOME_API_KEY="your_key_here"

# Run the full pipeline
lsd-alpha run \
  --gene-bed demo/input/mes_gbm_genes.hg19.bed \
  --tf-list demo/input/mes_gbm_tfs_mgt1-2.txt \
  --motifs demo/input/mes_motifs.meme \
  --ctcf-bed demo/input/ctcf_grch37_chr7_serpine1.bed \
  --fasta demo/input/chr7_region.fa \
  --cell-type "glioblastoma mesenchymal" \
  --out-dir results/

# Validate config only
lsd-alpha validate-config config.yaml
```

## Real ISM demo (SERPINE1)

```bash
export ALPHAGENOME_API_KEY="your_key_here"
python demo/run_real_alphagenome.py
# → demo/output/serpine1_ism_results.csv   (300 variants)
# → demo/output/top20_variants.tsv
# → demo/output/serpine1_wt_cre.txt
```

## Key result

From 300 saturation variants at `chr7:100,546,778–100,546,878` (hg38):
- **Best gain-of-function:** A>G at 100,546,785 (ATAC +0.65 log2FC)
- **Most disruptive:** T>G at 100,546,848 (ATAC −0.33 log2FC)

## References

- Gargiulo G et al. (2024). *Nature Communications* — LSD framework  
  [gitlab.com/gargiulo_lab/LSD_framework](https://gitlab.com/gargiulo_lab/LSD_framework)
- AlphaGenome API — Google DeepMind
- JASPAR 2024 — TF motif database
- MOODS-python — motif scanning library
