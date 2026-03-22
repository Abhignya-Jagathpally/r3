process INTEGRATION {
    """Batch effect correction and integration"""

    tag "integration"
    label "gpu"

    input:
        path data
        path config

    output:
        path "data/integrated/integrated.h5ad", emit: data

    script:
    """
    python -m src --stage integrate \
        --config ${config} \
        --data-dir data
    """
}
