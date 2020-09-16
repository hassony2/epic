from libyana.visutils import detect2d
from epic import boxutils


def add_boxesgt_viz(ax, obj_df, resize_factor=1, debug=False):
    if obj_df.shape[0] > 0:
        if debug:
            print("Drawing ground truth object box !")
        boxes = obj_df.box.values
        labels = obj_df.noun.values
        bboxes_norm = [
            boxutils.epic_box_to_norm(bbox, resize_factor=resize_factor)
            for bbox in boxes
        ]
        label_color = "w"
        detect2d.visualize_bboxes(
            ax, bboxes_norm, labels=labels, label_color=label_color, linewidth=2
        )
