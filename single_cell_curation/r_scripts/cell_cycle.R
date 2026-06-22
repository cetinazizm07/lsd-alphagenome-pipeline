suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(Matrix))

parser <- ArgumentParser(description = "Seurat CellCycleScoring — S and G2M scores per cell")
parser$add_argument("--counts-rds", required = TRUE, dest = "counts_rds",
    help = "Path to sparse counts matrix RDS (genes x cells)")
parser$add_argument("--s-genes-rds", required = TRUE, dest = "s_genes_rds",
    help = "Path to RDS file containing S-phase gene list (character vector)")
parser$add_argument("--g2m-genes-rds", required = TRUE, dest = "g2m_genes_rds",
    help = "Path to RDS file containing G2M-phase gene list (character vector)")
parser$add_argument("--output-tsv", required = TRUE, dest = "output_tsv",
    help = "Output TSV: cell_barcode, S.Score, G2M.Score, Phase")
args <- parser$parse_args()

counts   <- readRDS(args$counts_rds)
s_genes  <- readRDS(args$s_genes_rds)
g2m_genes <- readRDS(args$g2m_genes_rds)

so <- CreateSeuratObject(counts = counts)
so <- NormalizeData(so, verbose = FALSE)
so <- CellCycleScoring(so, s.features = s_genes, g2m.features = g2m_genes, set.ident = FALSE)

out <- data.frame(
    cell_barcode = rownames(so@meta.data),
    S.Score      = so@meta.data$S.Score,
    G2M.Score    = so@meta.data$G2M.Score,
    Phase        = so@meta.data$Phase
)
write.table(out, args$output_tsv, sep = "\t", row.names = FALSE, quote = FALSE)
cat("Cell cycle scoring done. Wrote:", args$output_tsv, "\n")
