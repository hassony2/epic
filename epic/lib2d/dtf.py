import cv2
import numpy as np


def distance_transform(mask, mask_size=5, pixel_scaling=200):
    mask_np = mask.cpu().detach().numpy().astype(np.uint8)
    dtf = cv2.distanceTransform(
        1 - mask_np, distanceType=cv2.DIST_L2, maskSize=mask_size
    )
    return 1 - dtf / pixel_scaling
