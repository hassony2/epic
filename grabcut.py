import cv2
import numpy as np

from epic.viz import boxviz

frame_idx = 300
img_path = f"/sequoia/data2/dataset/epic-100/EPIC-KITCHENS/P01/rgb_frames/P01_101/frame_{frame_idx:010d}.jpg"
im = cv2.imread(img_path)
if im is None:
    raise ValueError(f"Could not load image from {img_path}")
bboxes = [[200, 120, 230, 150]]


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


for iter_count in [4]:
    masks = []
    for bbox in bboxes:
        print(bbox)
        mask, masked = grab_cut(im, bbox, iter_counts=iter_count)
        masks.append(mask)

    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    ax = axes[0]
    ax.imshow(im[:, :, ::-1])
    ax.axis("off")
    ax = axes[1]
    ax.imshow(im[:, :, ::-1])
    for mask in masks:
        ax.imshow(mask, alpha=0.3)

    boxviz.add_bboxes(ax, bboxes)
    # ax.imshow(im[:, :, ::-1])
    ax.imshow(np.sum(masks, 0), alpha=0.3)

    fig.savefig(f"tmp{iter_count}.png")
import pdb

pdb.set_trace()
