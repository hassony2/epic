import cv2
import numpy as np

from epic.viz import boxviz
from epic.lib2d import grabcut

frame_idx = 300
img_path = (
    "/sequoia/data2/dataset/epic-100/EPIC-KITCHENS/"
    f"P01/rgb_frames/P01_101/frame_{frame_idx:010d}.jpg"
)
im = cv2.imread(img_path)
if im is None:
    raise ValueError(f"Could not load image from {img_path}")
bboxes = [[200, 120, 230, 150]]


for iter_count in [4]:
    masks = []
    for bbox in bboxes:
        print(bbox)
        mask, masked = grabcut.grab_cut(im, bbox, iter_counts=iter_count)
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
