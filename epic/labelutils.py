import numpy as np
import cv2


def extend_action_labels(video_action_data):
    dense_annots = {}
    for row_idx, action_row in video_action_data.iterrows():
        start_frame = action_row["start_frame"]
        stop_frame = action_row["stop_frame"]

        narration = action_row["narration"]
        # all_nouns = action_row['all_nouns']
        # noun = action_row['noun']
        # verb = action_row['verb']
        for frame_idx in range(start_frame, stop_frame + 1):
            dense_annots[frame_idx] = narration
    return dense_annots


def get_annot_adv(
    frame_idx,
    action_labels,
    cmapping,
    span=100,
    extent=7,
    bg_color=(0, 0, 0, 1),
    current_color=(1, 1, 1, 1),
    vert_ratio=10,
):
    """
    Args:
        extent (int): controls the height and width of the progress bar
        span (int): temporal span of action progress bar
    """
    colors = []
    switch_label_idxs = {}
    # For frame index get action color
    current_action = None
    for col_idx, adv_idx in enumerate(range(frame_idx - span, frame_idx + span)):
        if adv_idx in action_labels:
            action = action_labels[adv_idx]
            color = cmapping[action]
            if action != current_action:
                switch_label_idxs[col_idx] = action
            current_action = action
        else:
            current_action = None
            action = None
            color = bg_color
        # Add current timestamp mark
        if adv_idx == frame_idx:
            color = current_color

        colors.append(color)
    colors = np.array(
        [
            [
                color,
            ]
            * extent
            for color in colors
        ]
    ).reshape(len(colors) * extent, 4)
    colors = (
        colors[:, :3].reshape((-1, *colors[:, :3].shape)).repeat(extent * vert_ratio, 0)
    )
    thickness = 3
    # Anchor class label at class switch location
    for switch_col_idx, switch_label_class in switch_label_idxs.items():
        text_origin = ((switch_col_idx + 2) * extent, int(0.8 * extent * vert_ratio))
        colors = cv2.putText(
            colors,
            switch_label_class,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            # (255, 255, 255),
            (0, 0, 0),
            thickness,
        )
    return colors
