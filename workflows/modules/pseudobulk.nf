process PSEUDOBULK {
    """Pseudobulk analysis"""

    tag "pseudobulk"
    label "cpu"

    input:
        path data
        path annotations
        path config

    output:
        path "results/pseudobulk/pseudobulk.parquet", emit: data

    script:
    """
    python -m src --stage pseudobulk \
        --config ${config} \
        --data-dir data
    """
}
