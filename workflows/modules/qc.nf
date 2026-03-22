process QC {
    """Quality control and preprocessing"""

    tag "qc"
    label "gpu"

    input:
        path data
        path config

    output:
        path "data/standardized/*.h5ad", emit: data

    script:
    """
    python -m src --stage preprocess \
        --config ${config} \
        --data-dir data
    """
}
