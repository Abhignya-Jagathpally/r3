process EVALUATION {
    """Evaluation of clustering and annotation"""

    tag "evaluation"
    label "cpu"

    input:
        path data
        path annotations
        path config

    output:
        path "results/evaluation/metrics.json", emit: metrics

    script:
    """
    python -m src --stage evaluate \
        --config ${config} \
        --data-dir data
    """
}
