from collections import defaultdict

import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from libyana.metrics.iou import get_iou


def dfbox_to_norm(row, resize_factor=1):
    box = [
        row.left * resize_factor,
        row.top * resize_factor,
        row.right * resize_factor,
        row.bottom * resize_factor,
    ]
    return box


def epic_box_to_norm(bbox, resize_factor=1):
    bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
    bbox = [
        bbox[0] * resize_factor,
        bbox[1] * resize_factor,
        (bbox[0] + bbox[2]) * resize_factor,
        (bbox[1] + bbox[3]) * resize_factor,
    ]
    return bbox


def get_center_scale(bbox, scale_factor=1):
    center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * scale_factor
    return center, scale


def get_closest(value, candidates):
    val_dists = np.abs(candidates - value)
    dist = val_dists.min()
    return (dist, candidates[val_dists.argmin()])


def interpolate_bboxes(start_frame, end_frame, start_bbox, end_bbox):
    inter_boxes = {}
    all_inter_vals = []
    for start_val, end_val in zip(start_bbox, end_bbox):
        inter_vals = np.linspace(start_val, end_val, int(end_frame - start_frame))
        all_inter_vals.append(inter_vals)
    all_inter_vals = np.array(all_inter_vals).transpose()
    for step_idx in range(0, int(end_frame - start_frame) - 1):
        frame_idx = step_idx + start_frame + 1
        inter_boxes[frame_idx] = all_inter_vals[step_idx]
    return inter_boxes


def extend_labels(action_data):
    start_frames = action_data.start_frame.values
    stop_frames = action_data.stop_frame.values
    narrations = action_data.narration.values
    label_nb = len(narrations)
    all_annots = {}
    for label_idx in range(label_nb):
        start_frame = start_frames[label_idx]
        stop_frame = stop_frames[label_idx]
        narration = narrations[label_idx]
        for frame_idx in range(start_frame, stop_frame + 1):
            all_annots[frame_idx] = narration
    return all_annots


def extend_props(candidates, interpolate=True, max_frame_dist=50):
    """
    Args:
        max_frame_dist (int): dist which defines distance between consecutive
            annotations (30 in this case)
    """
    # Only keep frames with non-empty bbox annotations
    candidates = candidates[candidates.bounding_boxes.map(lambda d: len(d)) > 0]
    annot_frames = candidates["action_frame"].values.astype(np.int)
    # annot_frames = candidates['frame'].values.astype(np.int)
    annot_nouns = candidates["noun"].values
    annot_vids = candidates["video_id"].values
    extended_bboxes = []
    track_idx_map = dict()
    track_count = 0
    for annot_idx, (annot_frame, annot_noun, annot_vid) in enumerate(
        tqdm(zip(annot_frames[:-1], annot_nouns[:-1], annot_vids[:-1]))
    ):
        next_frame = annot_frames[annot_idx + 1]
        next_noun = annot_nouns[annot_idx + 1]
        annot_candidates = candidates[candidates.video_id == annot_vid][
            candidates.action_frame == annot_frame
        ][candidates.noun == annot_noun]
        annot_bboxes_candidates = annot_candidates.bounding_boxes.values
        annot_boxes = [boxes[0] for boxes in annot_bboxes_candidates]
        next_candidates = candidates[candidates.action_frame == next_frame][
            candidates.noun == annot_noun
        ][candidates.video_id == annot_vid]
        if next_candidates.bounding_boxes.shape[0] > 0:
            next_bboxes_candidates = next_candidates.bounding_boxes.values
            next_boxes = [boxes[0] for boxes in next_bboxes_candidates]

            # Detect consecutive annots with same nouns
            box_key = (annot_frame, annot_noun, annot_vid)
            track_idxs = []
            for box_idx in range(len(annot_boxes)):
                # Get current track index
                if box_key in track_idx_map:
                    track_idx = track_idx_map[box_key][box_idx]
                else:
                    track_idx = track_count
                    track_count += 1
                track_idxs.append(track_idx)
            track_idx_map[box_key] = track_idxs
            for track_idx, box in zip(track_idxs, annot_boxes):
                extended_bboxes.append(
                    {
                        "frame": annot_frame,
                        "box": box,
                        "noun": annot_noun,
                        "track_idx": track_idx,
                        "video_id": annot_vid,
                        "interp_annot": False,
                    }
                )
            # Detects both changes of videos with decrement of frame_idx and skipped frames
            if (
                np.abs(next_frame - annot_frame) < max_frame_dist
                and next_noun == annot_noun
            ):
                ious = np.zeros((len(annot_boxes), len(next_boxes)))
                for cur_idx, annot_box in enumerate(annot_boxes):
                    for next_idx, next_box in enumerate(next_boxes):
                        ious[cur_idx, next_idx] = get_iou(
                            epic_box_to_norm(annot_box), epic_box_to_norm(next_box)
                        )

                # Assign track indexes favoring large ious
                cur_idxs, next_idxs = linear_sum_assignment(-ious)
                cur_idxs, next_idxs = cur_idxs.tolist(), next_idxs.tolist()
                next_track_idxs = []
                for idx in range(len(next_boxes)):
                    if idx in next_idxs:
                        next_corresp_idx = next_idxs.index(idx)
                        cur_corresp_idx = cur_idxs[next_corresp_idx]
                        track_idx = track_idxs[cur_corresp_idx]
                    else:
                        track_idx = track_count
                        track_count += 1
                    next_track_idxs.append(track_idx)
                all_inter_boxes = []
                if interpolate:
                    for idx, track_idx in enumerate(track_idxs):
                        if idx in next_idxs:
                            next_idx = next_idxs.index(idx)
                            annot_idx = cur_idxs[next_corresp_idx]
                            inter_bboxes = interpolate_bboxes(
                                annot_frame,
                                next_frame,
                                annot_boxes[annot_idx],
                                next_boxes[next_idx],
                            )
                            all_inter_boxes.append((track_idx, inter_bboxes))
                    # Recover intermediate frame indices
                    for track_idx, bboxes in all_inter_boxes:
                        for frame in all_inter_boxes[0][1].keys():
                            box_key = (frame, annot_noun, annot_vid)
                            track_idx_map[box_key] = next_track_idxs
                            extended_bboxes.append(
                                {
                                    "frame": frame,
                                    "box": bboxes[frame],
                                    "noun": annot_noun,
                                    "track_idx": track_idx,
                                    "video_id": annot_vid,
                                    "interp_annot": True,
                                }
                            )
                    box_key = (next_frame, annot_noun, annot_vid)
                    track_idx_map[box_key] = next_track_idxs
    return extended_bboxes
