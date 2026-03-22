#!/usr/bin/env nextflow

/*
 * R3-MM Pipeline: Multiple Myeloma Single-Cell Computational Biology
 * Main Nextflow workflow
 */

nextflow.enable.dsl = 2

include { DOWNLOAD } from './modules/download'
include { QC } from './modules/qc'
include { INTEGRATION } from './modules/integration'
include { ANNOTATION } from './modules/annotation'
include { PSEUDOBULK } from './modules/pseudobulk'
include { EVALUATION } from './modules/evaluation'

/*
 * Pipeline parameters
 */
params.help = false
params.version = false
params.config = './configs/pipeline_config.yaml'
params.outdir = './results'
params.skip_download = false
params.skip_qc = false
params.skip_integration = false
params.skip_annotation = false
params.skip_evaluation = false

/*
 * Print help message
 */
if (params.help) {
    log.info """
    R3-MM Pipeline: Multiple Myeloma Single-Cell Computational Biology
    ================================================================

    Usage:
        nextflow run workflows/main.nf --config configs/pipeline_config.yaml

    Options:
        --config FILE               Pipeline configuration file
        --outdir DIR                Output directory (default: ./results)
        --skip_download             Skip data download
        --skip_qc                   Skip QC and preprocessing
        --skip_integration          Skip integration
        --skip_annotation           Skip annotation
        --skip_evaluation           Skip evaluation
        --help                      Print this help message
        --version                   Print pipeline version

    Example:
        nextflow run workflows/main.nf \\
            --config configs/pipeline_config.yaml \\
            --outdir results \\
            -profile docker
    """.stripIndent()
    exit 0
}

if (params.version) {
    println "R3-MM Pipeline version 0.1.0"
    exit 0
}

/*
 * Validate parameters
 */
if (!file(params.config).exists()) {
    exit 1, "Config file not found: ${params.config}"
}

/*
 * Create output directory
 */
outdir = file(params.outdir)
if (!outdir.exists()) {
    outdir.mkdirs()
}

/*
 * Log pipeline information
 */
log.info """\
    ╔════════════════════════════════════════════════════════════╗
    ║     R3-MM Pipeline: Multiple Myeloma Analysis Pipeline     ║
    ║                     Version 0.1.0                          ║
    ╚════════════════════════════════════════════════════════════╝

    Config file   : ${params.config}
    Output dir    : ${params.outdir}
    Skip download : ${params.skip_download}
    Skip QC       : ${params.skip_qc}
    Skip integration: ${params.skip_integration}
    Skip annotation : ${params.skip_annotation}
    Skip evaluation : ${params.skip_evaluation}
    Profile       : ${workflow.profile}
    """.stripIndent()

/*
 * Main workflow
 */
workflow {
    main:
        if (!params.skip_download) {
            DOWNLOAD(params.config)
            qc_input = DOWNLOAD.out.data
        } else {
            qc_input = channel.fromPath("data/raw/*.h5ad")
        }

        if (!params.skip_qc) {
            QC(qc_input, params.config)
            integration_input = QC.out.data
        } else {
            integration_input = channel.fromPath("data/standardized/*.h5ad")
        }

        if (!params.skip_integration) {
            INTEGRATION(integration_input, params.config)
            annotation_input = INTEGRATION.out.data
        } else {
            annotation_input = channel.fromPath("data/analysis_ready/*.h5ad")
        }

        if (!params.skip_annotation) {
            ANNOTATION(annotation_input, params.config)
            eval_input = ANNOTATION.out.data
        } else {
            eval_input = channel.fromPath("data/analysis_ready/*.h5ad")
        }

        if (!params.skip_evaluation) {
            EVALUATION(eval_input, ANNOTATION.out.annot, params.config)
        }

        if (!params.skip_evaluation) {
            PSEUDOBULK(eval_input, ANNOTATION.out.annot, params.config)
        }

    emit:
        results = EVALUATION.out.metrics
}

/*
 * Completion handlers
 */
workflow.onComplete {
    println """
    ╔════════════════════════════════════════════════════════════╗
    ║                   Pipeline Execution Summary               ║
    ╚════════════════════════════════════════════════════════════╝

    Status        : ${workflow.success ? 'SUCCESS' : 'FAILED'}
    Start time    : ${workflow.start}
    End time      : ${workflow.complete}
    Duration      : ${workflow.duration}
    Exit status   : ${workflow.exitStatus}
    """.stripIndent()

    if (workflow.success) {
        log.info "Results saved to: ${params.outdir}"
    } else {
        log.error "Pipeline failed! Check logs for details."
    }
}

workflow.onError {
    log.error "Pipeline execution stopped with the following message:"
    log.error workflow.errorMessage
}
