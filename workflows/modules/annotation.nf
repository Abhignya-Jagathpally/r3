process ANNOTATION {
    """Cell type annotation"""

    tag "annotation"
    label "gpu"

    input:
        path data
        path config

    output:
        path "data/annotated/annotated.h5ad", emit: data
        path "data/annotated/celltypes.csv", emit: annot

    script:
    """
    python -m src --stage annotate \
        --config ${config} \
        --data-dir data
    """
}
