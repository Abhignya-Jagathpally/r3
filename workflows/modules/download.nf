process DOWNLOAD {
    """Download datasets from GEO"""

    tag "download"
    label "gpu"

    input:
        path config

    output:
        path "data/raw/*.h5ad", emit: data

    script:
    """
    python -m src --stage download \
        --config ${config} \
        --data-dir data
    """
}
