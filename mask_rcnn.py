import cv2
import torch

from detectron2.data import transforms
from detectron2 import config as detcfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances, Boxes

from epic.masks import bboxmasks
from epic.viz import boxviz

mask_extractor = bboxmasks.MaskExtractor()
frame_idx = 300
img_path = f"/sequoia/data2/dataset/epic-100/EPIC-KITCHENS/P01/rgb_frames/P01_101/frame_{frame_idx:010d}.jpg"
im = cv2.imread(img_path)
if im is None:
    raise ValueError(f"Could not load image from {img_path}")
bboxes = [[0, 0, 100, 100], [100, 100, 200, 200], [100, 000, 500, 600]]
bboxes = [[100, 000, 500, 600]]
all_masks = []
pred_classes = list(range(1))
bboxes = [bboxes[0] for _ in pred_classes]
# res = mask_extractor.img_inference(im)
res = mask_extractor.masks_from_bboxes(im, boxes=bboxes, pred_classes=pred_classes)
masks = res["masks"]
boxes = res["boxes"]
scores = res["scores"]
classes = res["classes"]
labels = [
    f"{lab}: {score:.2f}" for score, lab in zip(res["scores"].cpu().detach(), classes)
]

from matplotlib import pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 10))
ax = axes[0]
ax.imshow(im[:, :, ::-1])
ax.axis("off")
ax = axes[1]
ax.imshow(im[:, :, ::-1])
# for mask in masks:
#     ax.imshow(mask.cpu().detach().numpy(), alpha=0.3)

boxviz.add_bboxes(ax, boxes.cpu().detach().numpy(), labels=labels)
# ax.imshow(im[:, :, ::-1])
ax.imshow(masks.cpu().detach().numpy().sum(0), alpha=0.3)

fig.savefig("tmp.png")
import pdb

pdb.set_trace()
