suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(SoupX))
suppressPackageStartupMessages(library(Matrix))

parser <- ArgumentParser(description = "SoupX ambient RNA removal")
parser$add_argument("--filtered", required = TRUE,
    help = "Path to filtered matrix directory (barcodes.tsv, features.tsv, matrix.mtx)")
parser$add_argument("--raw", required = TRUE,
    help = "Path to CellRanger raw (unfiltered) matrix directory")
parser$add_argument("--output-dir", required = TRUE, dest = "output_dir",
    help = "Output directory for corrected counts matrix (10x MTX format)")
args <- parser$parse_args()

read_10x_dir <- function(dir_path) {
    mtx_file  <- list.files(dir_path, pattern = "matrix\\.mtx",    full.names = TRUE, recursive = FALSE)[1]
    feat_file <- list.files(dir_path, pattern = "features|genes",  full.names = TRUE, recursive = FALSE)[1]
    bc_file   <- list.files(dir_path, pattern = "barcodes\\.tsv",  full.names = TRUE, recursive = FALSE)[1]
    mat      <- readMM(mtx_file)
    features <- read.table(feat_file, header = FALSE, sep = "\t", stringsAsFactors = FALSE)
    barcodes <- readLines(bc_file)
    rownames(mat) <- features[, 1]
    colnames(mat) <- barcodes
    as(mat, "CsparseMatrix")
}

tod <- read_10x_dir(args$raw)
toc <- read_10x_dir(args$filtered)

sc <- SoupChannel(tod, toc)
sc <- autoEstCont(sc)
corrected <- adjustCounts(sc)

dir.create(args$output_dir, showWarnings = FALSE, recursive = TRUE)
writeMM(corrected, file.path(args$output_dir, "matrix.mtx"))
write.table(
    data.frame(gene_id = rownames(corrected), gene_name = rownames(corrected),
               feature_type = "Gene Expression"),
    file.path(args$output_dir, "features.tsv"),
    sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE
)
writeLines(colnames(corrected), file.path(args$output_dir, "barcodes.tsv"))
cat("SoupX done. Corrected matrix written to:", args$output_dir, "\n")
