import numpy as np


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


def get_annot_adv(frame_idx, action_labels, cmapping, span=100, extent=7):
    """
    Args:
        extent (int): controls the height and width of the progress bar
        span (int): temporal span of action progress bar
    """
    colors = []
    # For frame index get action color
    for adv_idx in range(frame_idx - span, frame_idx + span):
        if adv_idx in action_labels:
            action = action_labels[adv_idx]
            color = cmapping[action]
        else:
            color = [0, 0, 0, 1]
        if adv_idx == frame_idx:
            color = [1, 1, 1, 1]
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
    colors = colors[:, :3].reshape((-1, *colors[:, :3].shape)).repeat(extent *
            10, 0)
    return colors
