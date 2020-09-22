from copy import deepcopy
from pathlib import Path
from epic.masks import io as masksio
import pandas as pd
from functools import lru_cache

import numpy as np
from PIL import Image


def framedet2dicts(det, obj_thresh=0.5, hand_thresh=0.5, height=1080, width=1920):
    res_dict = {"video_id": det.video_id, "frame": det.frame_number}
    dicts = []
    for obj_det in det.objects:
        det_dict = deepcopy(res_dict)
        score = obj_det.score
        if score > obj_thresh:
            det_dict["score"] = score
            det_dict["left"] = obj_det.bbox.left * width
            det_dict["right"] = obj_det.bbox.right * width
            det_dict["top"] = obj_det.bbox.top * height
            det_dict["bottom"] = obj_det.bbox.bottom * height
            det_dict["pred_class"] = obj_det.pred_class
            # det_dict["mask"] = resize_mask(obj_det.mask, height=height, width=width)
            det_dict["mask"] = obj_det._coco_mask_counts
            dicts.append(det_dict)
        dicts.append(det_dict)
    return dicts


@lru_cache(maxsize=128)
def load_video_masks(video_id, masks_root):
    """
    Args:
        video_id (str): PXX_XX video id
        masks_root (str): path to mask folder
                \ P01
                    \ P01_01.pkl
                \ ...
    """
    masks_root = Path(masks_root)
    masks_list = masksio.load_detections(masks_root / video_id[:3] / f"{video_id}.pkl")
    all_masks_dicts = []
    print(masks_root)
    for masks_det in masks_list:
        masks_dicts = framedet2dicts(masks_det)
        all_masks_dicts.extend(masks_dicts)
    dat = pd.DataFrame(all_masks_dicts)
    return dat
