suppressPackageStartupMessages(library(argparse))
suppressPackageStartupMessages(library(GEOquery))

parser <- ArgumentParser(description = "Download GEO dataset via GEOquery")
parser$add_argument("--accession", required = TRUE,
    help = "GEO accession ID (e.g. GSE84465)")
parser$add_argument("--staging-dir", required = TRUE, dest = "staging_dir",
    help = "Local directory to download files into")
args <- parser$parse_args()

dir.create(args$staging_dir, showWarnings = FALSE, recursive = TRUE)
options(timeout = 600)

tryCatch({
    gse <- getGEO(args$accession, destdir = args$staging_dir, GSEMatrix = TRUE)
    cat("Series matrix downloaded for", args$accession, "\n")
}, error = function(e) {
    cat("getGEO warning (non-fatal):", conditionMessage(e), "\n")
})

getGEOSuppFiles(args$accession, makeDirectory = FALSE, baseDir = args$staging_dir)
cat("GEOquery done. Supplementary files in:", args$staging_dir, "\n")
