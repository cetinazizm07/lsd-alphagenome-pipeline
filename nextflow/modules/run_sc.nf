process RUN_SC {
    tag "${meta.name}"
    publishDir "${params.outdir}/${meta.name}", mode: 'copy'

    input:
    tuple val(meta), path(config_yaml)

    output:
    tuple val(meta), path("*.h5ad"),       emit: h5ad
    tuple val(meta), path("qc_report.json"), emit: report
    tuple val(meta), path("*.png"),          emit: plots, optional: true

    script:
    """
    sc-curate run-sc --config '${config_yaml}' --out .
    """
}
