process RUN_BULK {
    tag "${meta.name}"
    publishDir "${params.outdir}/${meta.name}", mode: 'copy'

    input:
    tuple val(meta), path(config_yaml)

    output:
    tuple val(meta), path("*.h5ad"),        emit: h5ad
    tuple val(meta), path("qc_report.json"), emit: report

    script:
    """
    sc-curate run-bulk --config '${config_yaml}' --out .
    """
}
