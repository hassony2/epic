from libyana.visutils import detect2d


def add_hoa_viz(ax, hoa_df, resize_factor=1, debug=False):
    if hoa_df.shape[0] > 0:
        if debug:
            print("Drawing predicted hand and object boxes !")
        bboxes_norm = [
            [
                box_row[1].left * resize_factor,
                box_row[1].top * resize_factor,
                box_row[1].right * resize_factor,
                box_row[1].bottom * resize_factor,
            ]
            for box_row in hoa_df.iterrows()
        ]
        colors = [get_hoa_color(obj[1]) for obj in hoa_df.iterrows()]
        labels = [get_hoa_label(obj[1]) for obj in hoa_df.iterrows()]
        detect2d.visualize_bboxes(
            ax,
            bboxes_norm,
            labels=labels,
            label_color="w",
            linewidth=2,
            color=colors,
        )


def get_hoa_color(obj):
    if obj.det_type == "hand":
        if obj.side == "right":
            return "g"
        elif obj.side == "left":
            return "m"
    else:
        return "k"


def get_hoa_label(obj):
    if obj.det_type == "hand":
        hoa_label = obj.hoa_link[:5]
        if obj.side == "right":
            label = "hand_r" + hoa_label
        elif obj.side == "left":
            label = "hand_l" + hoa_label
        else:
            raise ValueError("hand side {obj.side} not in [left|right]")
    else:
        label = "obj"
    return f"{label}: {obj.score:.2f}"
