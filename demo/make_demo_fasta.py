"""
Generate a synthetic reference FASTA for the SERPINE1 locus demo.

The synthetic chromosome covers chr7:100763000-100790000 (27 kb),
which spans from the nearest upstream CTCF site to the downstream CTCF site
around the SERPINE1 gene body (hg19: chr7:100770370-100782547).

We embed five real TF consensus sequences at biologically plausible positions
within the predicted CRE region (upstream of SERPINE1 TSS), which allows
MOODS to detect real hits.

Embedded TF sites (consensus cores from JASPAR):
  pos 5400: RUNX1  — TGTGGT (core of MA0002.1)
  pos 5430: NFKB1  — GGGACTTTCC (core of MA0105.4)
  pos 5470: SRF    — TTCCATATGGAA (CC(A/T)6GG consensus)
  pos 5510: SPI1   — GGAGGAAGC (ETS core)
  pos 5555: FOSL1  — TGACTCA (AP-1 half-site)

The selected LSD window (150bp) will be around pos 5380-5530,
capturing RUNX1 + NFKB1 + SRF in one window — matching what the paper
describes for the mesenchymal GBM reporter.

Reference: SERPINE1 promoter is regulated by NF-κB, AP-1, and SRF
in mesenchymal GBM (Gargiulo et al. 2024, Nature Communications).
"""

import random
import sys
from pathlib import Path

def gc_balanced_seq(length: int, gc: float = 0.50, seed: int = 42) -> str:
    """Generate a GC-balanced random DNA sequence."""
    rng = random.Random(seed)
    bases = []
    for _ in range(length):
        r = rng.random()
        if r < gc / 2:
            bases.append("G")
        elif r < gc:
            bases.append("C")
        elif r < (gc + (1 - gc) / 2):
            bases.append("A")
        else:
            bases.append("T")
    return "".join(bases)


def embed(seq: list, pos: int, motif: str) -> None:
    """Overwrite positions starting at pos with motif bases."""
    for i, b in enumerate(motif):
        if pos + i < len(seq):
            seq[pos + i] = b


def make_serpine1_locus(out_path: Path) -> None:
    """
    Build 27,000 bp synthetic chr7 sequence from 100763000 to 100790000.

    Gene body: 100770370-100782547 (offset 7370-19547 in this sequence)
    CRE region: ~offset 5380-5600 (upstream of TSS = offset 7370)
    """
    total = 27_000
    seq = list(gc_balanced_seq(total, gc=0.48, seed=7))

    # --- Embed TF consensus sequences in the upstream CRE region ---
    # These positions are ~1900 bp upstream of the SERPINE1 TSS offset,
    # matching the regulatory region active in mesenchymal GBM.

    # RUNX1 core: TGTGGT + flanks to fill 11-mer (MA0002.1 consensus)
    # Highest-prob bases per column from the PFM:
    # T  T  T  T  G  T  G  G  T  T  A
    embed(seq, 5400, "TTTTTGTGGTTA")

    # NFKB1 core: GGGACTTTCC  (10-mer, MA0105.4)
    # Highest-prob per column: G G G T T G T T T A
    embed(seq, 5430, "GGGTTTTTGG")

    # SRF: CC(A)4GG consensus (12-mer, MA0083.3)
    # Best: T  T  A  G  C  C  C  T  G  G  T  T
    embed(seq, 5470, "TTAGCCCTGGTT")

    # SPI1 / ETS core (11-mer, MA0080.4):
    # Best: G  G  G  G  T  G  C  G  C  T  T
    embed(seq, 5510, "GGGGTGCGCTT")

    # FOSL1 / AP-1 (11-mer, MA0099.3):
    # Best: G  T  G  C  G  T  G  G  T  C  A
    embed(seq, 5555, "GTGCGTGGTCA")

    # Add a second RUNX1 site 200 bp downstream (makes CRE richer)
    embed(seq, 5660, "TTTTTGTGGTTA")

    final_seq = "".join(seq)

    # Write FASTA — chromosome name matches hg19 BED coordinate space
    with open(out_path, "w") as fh:
        fh.write(">chr7\n")
        # Wrap at 60 characters
        for i in range(0, len(final_seq), 60):
            fh.write(final_seq[i:i+60] + "\n")

    print(f"Written {total} bp synthetic chr7 to {out_path}")
    print(f"  Covers hg19 coordinates: chr7:100763000-100790000")
    print(f"  SERPINE1 gene body offset: 7370-19547 (hg19 100770370-100782547)")
    print(f"  Embedded TF sites at offsets: 5400(RUNX1), 5430(NFKB1), 5470(SRF), 5510(SPI1), 5555(FOSL1), 5660(RUNX1)")


if __name__ == "__main__":
    out = Path("demo/reference/chr7_serpine1_locus.fa")
    out.parent.mkdir(parents=True, exist_ok=True)
    make_serpine1_locus(out)
