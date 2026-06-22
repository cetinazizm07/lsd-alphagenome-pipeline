process DOWNLOAD {
    tag "${meta.name}"
    publishDir "${params.outdir}/${meta.name}/staging", mode: 'copy'

    input:
    tuple val(meta)

    output:
    tuple val(meta), path("run_config.yaml"), emit: samples

    script:
    def local_flag = meta.local_path ? "--input '${meta.local_path}'" : ""
    def accession_flag = meta.accession ? "--accession '${meta.accession}'" : ""
    """
    sc-curate download \
        --source '${meta.source}' \
        ${accession_flag} \
        ${local_flag} \
        --out .

    # Write a minimal config YAML for the next process
    cat > run_config.yaml <<CONFIG
dataset_name: ${meta.name}
modality: ${meta.modality}
source: ${meta.source}
geo_accession: "${meta.accession}"
local_data_path: "${meta.local_path}"
local_format: h5ad
output_dir: .
assembly: hg38
random_seed: 42
run_soupx: false
remove_doublets: true
score_cell_cycle: true
CONFIG
    """
}
