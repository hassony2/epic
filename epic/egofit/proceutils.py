import numpy as np
from scipy.spatial import distance_matrix


def preprocess_boxes(boxes):
    # Replace with closest
    empty_box_idxs = [idx for idx, box in enumerate(boxes) if not (len(box))]
    full_box_idxs = [idx for idx, box in enumerate(boxes) if len(box)]
    closest_idxs = distance_matrix(
        np.array(empty_box_idxs)[:, None], np.array(full_box_idxs)[:, None]
    ).argmin(1)
    fill_idxs = [full_box_idxs[idx] for idx in closest_idxs]
    for empty_idx, fill_idx in zip(empty_box_idxs, fill_idxs):
        boxes[empty_idx] = boxes[fill_idx]
    obj_bboxes = np.stack(boxes).transpose(1, 0, 2)  # (objects, time, 4)
    return obj_bboxes
