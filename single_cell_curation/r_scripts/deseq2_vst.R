suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(DESeq2))

parser <- ArgumentParser(description = "DESeq2 variance stabilizing transformation (VST)")
parser$add_argument("--counts-tsv", required = TRUE, dest = "counts_tsv",
    help = "Input TSV: rows=genes (with row names), columns=samples (with header)")
parser$add_argument("--output-tsv", required = TRUE, dest = "output_tsv",
    help = "Output TSV: VST-transformed values, same shape as input")
args <- parser$parse_args()

counts_mat <- read.table(args$counts_tsv, header = TRUE, row.names = 1,
                         sep = "\t", check.names = FALSE)
counts_mat <- round(as.matrix(counts_mat))
storage.mode(counts_mat) <- "integer"

col_data <- data.frame(
    row.names = colnames(counts_mat),
    condition = rep("A", ncol(counts_mat))
)
dds     <- DESeqDataSetFromMatrix(countData = counts_mat, colData = col_data, design = ~1)
vst_mat <- assay(vst(dds, blind = TRUE))

write.table(
    data.frame(gene = rownames(vst_mat), vst_mat, check.names = FALSE),
    args$output_tsv, sep = "\t", row.names = FALSE, quote = FALSE
)
cat("DESeq2 VST done. Wrote:", args$output_tsv, "\n")
