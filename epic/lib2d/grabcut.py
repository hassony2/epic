from copy import deepcopy

from matplotlib import pyplot as plt
import numpy as np
import cv2

from libyana.conversions import npt


def grab_cut(
    img,
    bbox=None,
    mask=None,
    bgd_mask=None,
    fgd_mask=None,
    iter_counts=5,
    margin=10,
    debug=True,
    fgd_model=None,
    bgd_model=None,
):
    if fgd_model is None:
        fgd_model = np.zeros((1, 65), np.float64)
    else:
        fgd_model = npt.numpify(fgd_model).astype(np.float64)
    if bgd_model is None:
        bgd_model = np.zeros((1, 65), np.float64)
    else:
        bgd_model = npt.numpify(bgd_model).astype(np.float64)
    if mask is not None:
        # ! bbox is not taken into account by cv2.grabCut !
        if bbox is None:
            xs, ys = np.where(mask)
            bbox = [xs.min(), ys.min(), xs.max(), ys.max()]
        # [x_min, y_min, x_max, y_max] --> [x_min, y_min, width, height]
        grab_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        # mask = (mask > 0).astype(np.uint8)
        #  mask = (np.where((mask != 0), cv2.GC_PR_FGD, cv2.GC_PR_BGD)).astype('uint8')
        mask = (np.where((mask != 0), cv2.GC_PR_FGD, cv2.GC_PR_BGD)).astype(
            "uint8"
        )
        if bgd_mask is not None:
            mask[bgd_mask > 0] = cv2.GC_BGD
        if fgd_mask is not None:
            mask[fgd_mask > 0] = cv2.GC_FGD
        init_mask = deepcopy(mask)
        cv2.grabCut(
            img,
            mask,
            grab_bbox,
            bgd_model,
            fgd_model,
            iter_counts,
            cv2.GC_INIT_WITH_MASK,
        )
        if debug:
            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            ax = axes[0]
            ax.axis("off")
            ax.imshow(img)
            ax = axes[1]
            ax.axis("off")
            ax.imshow(img)
            ax.imshow(init_mask, alpha=0.5)
            ax = axes[2]
            ax.axis("off")
            ax.imshow(img)
            ax.imshow(mask, alpha=0.5)
            fig.savefig("tmp.png", bbox_inches="tight")
    else:
        if margin:
            bbox = [
                max(0, bbox[0] - margin),
                max(0, bbox[1] - margin),
                min(img.shape[0], bbox[2] + margin),
                min(img.shape[1], bbox[3] + margin),
            ]
        mask = np.zeros(img.shape[:2], np.uint8)
        # [x_min, y_min, x_max, y_max] --> [x_min, y_min, width, height]
        grab_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        cv2.grabCut(
            img,
            mask,
            grab_bbox,
            bgd_model,
            fgd_model,
            iter_counts,
            cv2.GC_INIT_WITH_RECT,
        )
    processed_mask = np.where(
        (mask == cv2.GC_PR_FGD) | (mask == cv2.GC_FGD), 1, 0
    ).astype("uint8")
    img = img * processed_mask[:, :, np.newaxis]
    return {
        "in_mask": init_mask,
        "grab_mask": processed_mask,
        "fgd_model": fgd_model,
        "bgd_model": bgd_model,
    }
