import cv2
import numpy as np
from epic import boxutils


def grab_cut(img, bbox, iter_counts=5, margin=10):
    mask = np.zeros(img.shape[:2], np.uint8)
    if margin:
        bbox = [
            max(0, bbox[0] - margin),
            max(0, bbox[1] - margin),
            min(img.shape[0], bbox[2] + margin),
            min(img.shape[1], bbox[3] + margin),
        ]
    # [x_min, y_min, x_max, y_max] --> [x_min, y_min, width, height]
    grab_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        img, mask, grab_bbox, bgd_model, fgd_model, iter_counts, cv2.GC_INIT_WITH_RECT
    )
    processed_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    img = img * processed_mask[:, :, np.newaxis]
    return processed_mask, img


def masks_from_df(img, df, input_format="RGB", resize_factor=1):
    obj_df = df[df.det_type == "object"]
    boxes = []
    masks = []
    if len(obj_df):
        for _, row in obj_df.iterrows():
            box = boxutils.dfbox_to_norm(row, resize_factor=resize_factor)
            mask, _ = grab_cut(img, box)
            masks.append(mask)
            boxes.append(box)
    return masks, boxes
