import os
from copy import deepcopy
from pathlib import Path
import pickle

import cv2
import numpy as np
import pandas as pd

from epic.boxutils import extend_props


def get_obj_labels(
    epic_annot_root="",
    split="train",
    video_id=None,
    person_id=None,
    use_cache=True,
    cache_folder="results/cache",
    interpolate=True,
):
    os.makedirs(cache_folder, exist_ok=True)

    if interpolate:
        interp_str = "inter"
    else:
        interp_str = "no_inter"
    cache_path = Path(cache_folder) / f"{video_id}_{person_id}_{split}_{interp_str}.pkl"
    if use_cache and os.path.exists(cache_path):
        print(f"Loading tracked object labels from cache {cache_path}")
        extended_obj_data = pd.read_pickle(cache_path)
    else:
        epic_annot_root = Path(epic_annot_root)
        with open(
            epic_annot_root / f"EPIC_{split}_object_action_correspondence.pkl", "rb"
        ) as p_f:
            corresp_data = pickle.load(p_f)
            corresp_data = corresp_data.reset_index()
        # Get object infos
        object_data = pd.read_csv(epic_annot_root / "EPIC_train_object_labels.csv")
        object_data.bounding_boxes = object_data.bounding_boxes.apply(eval)

        # Map to action frame indices
        object_data = pd.merge(
            object_data,
            corresp_data,
            how="left",
            left_on=["video_id", "frame", "participant_id"],
            right_on=["video_id", "object_frame", "participant_id"],
        )
        if video_id is not None:
            object_data = object_data[object_data["video_id"] == video_id]
        if person_id is not None:
            object_data = object_data[object_data.participant_id == person_id]
        extended_obj_data = pd.DataFrame(
            extend_props(object_data, interpolate=interpolate)
        )
        extended_obj_data.to_pickle(cache_path)
    return extended_obj_data


def extend_action_labels(video_action_data):
    dense_annots = {}
    dense_df = []
    for row_idx, (_, action_row) in enumerate(video_action_data.iterrows()):
        start_frame = action_row["start_frame"]
        stop_frame = action_row["stop_frame"]

        narration = action_row["narration"]
        # all_nouns = action_row['all_nouns']
        # noun = action_row['noun']
        # verb = action_row['verb']
        # Convert row to dict
        for frame_idx in range(start_frame, stop_frame + 1):
            annot_dict = deepcopy(action_row.to_dict())
            dense_annots[frame_idx] = narration
            annot_dict["frame_idx"] = frame_idx
            annot_dict["action_idx"] = row_idx
            dense_df.append(annot_dict)
    return dense_annots, pd.DataFrame(dense_df)


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
