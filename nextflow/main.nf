nextflow.enable.dsl = 2

include { DOWNLOAD } from './modules/download'
include { RUN_SC }   from './modules/run_sc'
include { RUN_BULK } from './modules/run_bulk'

params.samplesheet = "samplesheet.csv"
params.outdir      = "results"

workflow {
    ch_samples = Channel
        .fromPath(params.samplesheet)
        .splitCsv(header: true)
        .map { row ->
            def meta = [
                name:      row.name,
                source:    row.source,
                accession: row.accession ?: "",
                modality:  row.modality,
                local_path: row.local_path ?: "",
            ]
            tuple(meta)
        }

    DOWNLOAD(ch_samples)

    sc_ch   = DOWNLOAD.out.samples.filter { meta, config -> meta.modality == "single_cell" }
    bulk_ch = DOWNLOAD.out.samples.filter { meta, config -> meta.modality == "bulk" }

    RUN_SC(sc_ch)
    RUN_BULK(bulk_ch)
}
