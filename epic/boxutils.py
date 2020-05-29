from collections import defaultdict

import numpy as np

from libyana.metrics.iou import get_iou


def epic_box_to_norm(bbox, resize_factor=1):
    bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
    bbox = [
        bbox[0] * resize_factor,
        bbox[1] * resize_factor, (bbox[0] + bbox[2]) * resize_factor,
        (bbox[1] + bbox[3]) * resize_factor
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
        inter_vals = np.linspace(start_val, end_val, end_frame - start_frame)
        all_inter_vals.append(inter_vals)
    all_inter_vals = np.array(all_inter_vals).transpose()
    for step_idx in range(0, end_frame - start_frame - 1):
        frame_idx = step_idx + start_frame + 1
        inter_boxes[frame_idx] = (all_inter_vals[step_idx])
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


def extend_props(candidates, interpolate=True):
    annot_frames = candidates['frame'].values
    annot_nouns = candidates['noun'].values
    extended_bboxes = defaultdict(list)
    for annot_idx, (annot_frame, annot_noun) in enumerate(
            zip(annot_frames[:-1], annot_nouns[:-1])):
        next_frame = annot_frames[annot_idx + 1]
        next_noun = annot_nouns[annot_idx + 1]
        annot_candidates = candidates[candidates.frame == annot_frame][
            candidates.noun == annot_noun]
        annot_boxes = annot_candidates.bounding_boxes.values[0]
        # if len(annot_boxes) > 1:
        #     raise ValueError('TODO handle several boxes')
        next_candidates = candidates[candidates.frame == next_frame][
            candidates.noun == annot_noun]
        if next_candidates.bounding_boxes.shape[0] > 0:
            next_boxes = next_candidates.bounding_boxes.values[0]

            # Detect consecutive annots with same nouns
            extended_bboxes[annot_frame].append((annot_boxes, annot_noun))
            if next_frame - annot_frame < 31 and next_noun == annot_noun:
                ious = np.zeros((len(annot_boxes), len(next_boxes)))
                for cur_idx, annot_box in enumerate(annot_boxes):
                    for next_idx, next_box in enumerate(next_boxes):
                        ious[cur_idx,
                             next_idx] = get_iou(epic_box_to_norm(annot_box),
                                                 epic_box_to_norm(next_box))
                best_next_match_idxs = ious.argmax(1)
                all_inter_boxes = []
                if interpolate:
                    for annot_idx, annot_box in enumerate(annot_boxes):
                        inter_bboxes = interpolate_bboxes(
                            annot_frame, next_frame, annot_box,
                            next_boxes[best_next_match_idxs[annot_idx]])
                        all_inter_boxes.append(inter_bboxes)
                    for key in all_inter_boxes[0].keys():
                        extended_bboxes[key].append(([
                            inter_boxes[key] for inter_boxes in all_inter_boxes
                        ], annot_noun))
    return extended_bboxes
