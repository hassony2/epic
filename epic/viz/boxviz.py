from libyana.visutils import detect2d


def add_bboxes(ax, boxes, labels=None, colors=None):
    if colors is None:
        colors = ["w" for _ in boxes]
    if labels is None:
        labels = [str(idx) for idx in range(len(boxes))]
    detect2d.visualize_bboxes(
        ax,
        boxes,
        labels=labels,
        label_color="w",
        linewidth=2,
        color=colors,
    )
