import cv2
import numpy as np
import torch
import torchvision


def crops(images, boxes, output_size=None):
    assert images.dim() in [3, 4]
    if images.dim() == 3:
        images = images.unsqueeze(1)
    batch_size, _, h, w = images.shape
    device = images.device
    boxes = torch.cat(
        (torch.arange(batch_size).unsqueeze(1).to(device).float(), boxes),
        dim=1,
    )
    crops = torchvision.ops.roi_align(
        images, boxes, output_size=output_size, sampling_ratio=4
    )
    return crops


def crop_cv2(img, bbox, xy2yx=True, resize=None):
    x1, y1, x2, y2 = bbox
    if xy2yx:
        x1, y1, x2, y2 = y1, x1, y2, x2
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    crop_img = img[y1:y2, x1:x2]
    if resize is not None:
        crop_img = cv2.resize(crop_img.astype(np.float), resize)
    return crop_img


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    img = cv2.copyMakeBorder(
        img,
        -min(0, y1),
        max(y2 - img.shape[0], 0),
        -min(0, x1),
        max(x2 - img.shape[1], 0),
        cv2.BORDER_REPLICATE,
    )
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2
