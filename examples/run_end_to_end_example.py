"""
End-to-end working example of LSD x AlphaGenome Pipeline.

Demonstrates:
  1. Input definition (Gene BED, CTCF boundaries, MEME motifs, Reference FASTA)
  2. LSD CRE Selection (finds optimal 150 bp window by combinatorial TF density)
  3. In Silico Saturation Mutagenesis (protects locked TFBS; mutates others to 3 alts)
  4. AlphaGenome Scoring (predicts ATAC-seq accessibility + RNA expression log2FC)
  5. Ranking by Harmonic-Mean Fitness & Export (CSV, JSON, FASTA)
"""

import sys
from pathlib import Path
import pandas as pd

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lsd_alphagenome_pipeline.config import PipelineConfig
from lsd_alphagenome_pipeline.pipeline import Pipeline


def setup_example_data(data_dir: Path) -> dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Signature Gene BED (SERPINE1 at chr7:10000-15000)
    gene_bed = data_dir / "signature_genes.bed"
    gene_bed.write_text("chr7\t10000\t15000\tSERPINE1\t0\t+\n", encoding="utf-8")

    # 2. CTCF Boundary BED (CTCF loop anchors flanking the gene)
    ctcf_bed = data_dir / "ctcf_boundaries.bed"
    ctcf_bed.write_text(
        "chr7\t5000\t5400\tCTCF_UP\t0\t+\n"
        "chr7\t20000\t20400\tCTCF_DOWN\t0\t-\n",
        encoding="utf-8",
    )

    # 3. MEME Motif File (JASPAR motifs: RUNX1, NFKB1, SRF)
    meme_file = data_dir / "tf_motifs.meme"
    meme_content = """MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

MOTIF RUNX1
letter-probability matrix: alength= 4 w= 6 nsites= 20 E= 0
  0.05 0.05 0.05 0.85
  0.05 0.05 0.85 0.05
  0.05 0.05 0.05 0.85
  0.05 0.05 0.85 0.05
  0.05 0.05 0.85 0.05
  0.05 0.05 0.05 0.85

MOTIF NFKB1
letter-probability matrix: alength= 4 w= 10 nsites= 20 E= 0
  0.05 0.05 0.85 0.05
  0.05 0.05 0.85 0.05
  0.05 0.05 0.85 0.05
  0.85 0.05 0.05 0.05
  0.05 0.85 0.05 0.05
  0.05 0.05 0.05 0.85
  0.05 0.05 0.05 0.85
  0.05 0.05 0.05 0.85
  0.05 0.85 0.05 0.05
  0.05 0.85 0.05 0.05
"""
    meme_file.write_text(meme_content, encoding="utf-8")

    # 4. Reference FASTA (25 kb locus containing embedded motifs at promoter: chr7:8500-8650)
    fasta_file = data_dir / "locus_ref.fa"
    total_len = 25_000
    # Generate background DNA sequence
    seq_chars = list("ACGT" * (total_len // 4))

    # Embed RUNX1 (TGTGGT) at pos 8520
    runx1_motif = "TGTGGT"
    for i, b in enumerate(runx1_motif):
        seq_chars[8520 + i] = b

    # Embed NFKB1 (GGGACTTTCC) at pos 8550
    nfkb1_motif = "GGGACTTTCC"
    for i, b in enumerate(nfkb1_motif):
        seq_chars[8550 + i] = b

    seq_str = "".join(seq_chars)
    fasta_content = f">chr7\n"
    for i in range(0, len(seq_str), 60):
        fasta_content += seq_str[i : i + 60] + "\n"
    fasta_file.write_text(fasta_content, encoding="utf-8")

    return {
        "gene_bed": gene_bed,
        "ctcf_bed": ctcf_bed,
        "meme_file": meme_file,
        "fasta_file": fasta_file,
    }


def main():
    print("=" * 70)
    print(" LSD x AlphaGenome Pipeline — Uçtan Uca Örnek Çalıştırma ")
    print("=" * 70)

    work_dir = REPO_ROOT / "examples" / "workspace"
    input_dir = work_dir / "inputs"
    output_dir = work_dir / "outputs"

    print("\n[1/4] Girdi dosyaları hazırlanıyor...")
    inputs = setup_example_data(input_dir)
    print(f"  • Gen BED       : {inputs['gene_bed'].name}")
    print(f"  • CTCF BED      : {inputs['ctcf_bed'].name}")
    print(f"  • TF Motifleri  : {inputs['meme_file'].name} (RUNX1, NFKB1)")
    print(f"  • Referans FASTA: {inputs['fasta_file'].name} (chr7, 25,000 bp)")

    # Konfigürasyon
    config = PipelineConfig(
        genes_bed=inputs["gene_bed"],
        ctcf_bed=inputs["ctcf_bed"],
        tf_motifs=inputs["meme_file"],
        fasta=inputs["fasta_file"],
        output_dir=output_dir,
        assembly="hg38",
        cell_type="glioblastoma",
        strategy="saturation",
        motif_pvalue_threshold=0.001,
        alphagenome_api_key="mock",
        top_n_export=5,
        random_seed=42,
    )

    print("\n[2/4] Pipeline çalıştırılıyor (LSD -> Mutagenesis -> Scoring)...")
    pipeline = Pipeline(config)
    csv_result = pipeline.run()

    print("\n[3/4] Çalıştırma tamamlandı! Çıktılar üretildi:")
    print(f"  • Sonuç CSV     : {output_dir / 'results.csv'}")
    print(f"  • Sonuç JSON    : {output_dir / 'results.json'}")
    print(f"  • WT CRE FASTA  : {output_dir / 'wt_cre.fa'}")
    print(f"  • Konfigürasyon : {output_dir / 'run_config.yaml'}")

    print("\n[4/4] Çıktı Analizi ve Özet:")
    wt_fa = (output_dir / "wt_cre.fa").read_text(encoding="utf-8").strip()
    print("\n--- Seçilen Doğal WT CRE ---")
    print(wt_fa)

    print("\n--- En Yüksek Sıralı İlk 5 Varyant (Top 5 Mutants) ---")
    df = pd.read_csv(csv_result)
    print(df[["chrom", "pos_1based", "ref", "alt", "cre_offset", "atac_score", "rna_score", "log2fc_atac", "log2fc_rna", "fitness"]])

    print("\n" + "=" * 70)
    print(" Örnek çalıştırma başarıyla tamamlandı!")
    print("=" * 70)


if __name__ == "__main__":
    main()
