from matplotlib import cm
import numpy as np
from PIL import Image
import torch

from libyana.visutils import detect2d
from pycocotools.mask import decode as coco_mask_decode
from epic import boxutils


def resize_mask(
    mask: np.ndarray, height: int, width: int, smooth: bool = True
) -> np.ndarray:
    assert mask.ndim == 2
    if smooth:
        # The original masks seem to be
        mask_img = Image.fromarray(mask * 255)
        return (
            np.asarray(
                mask_img.resize((50, 50), Image.LANCZOS).resize(
                    (width, height), Image.LANCZOS
                )
            )
            > 128
        ).astype(np.uint8)
    return np.asarray(Image.fromarray(mask).resize((width, height), Image.NEAREST))


def add_masks_viz(ax, masks, bboxes_norm, labels=None, mask_alpha=0.6, debug=False):
    if labels is None:
        labels = ["coco" for _ in bboxes_norm]
    detect2d.visualize_bboxes(
        ax,
        bboxes_norm,
        labels=labels,
        label_color="w",
        linewidth=2,
        color="c",
    )
    for mask in masks:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().detach().numpy()
        base_mask = mask[:, :, np.newaxis].astype(np.float)
        base_mask = base_mask / base_mask.max()
        show_mask = np.concatenate(
            [
                # cm.YlGn(base_mask)[:, :, 0][:, :, :3],
                cm.hot(base_mask)[:, :, 0][:, :, :3],
                mask_alpha * (base_mask > 0).astype(np.float),
            ],
            2,
        )
        ax.imshow(show_mask)


def add_masks_df_viz(ax, masks_df, resize_factor=1, alpha_mask=0.5, debug=False):
    if masks_df.shape[0] > 0:
        if debug:
            print("Drawing predicted hand and object boxes !")
        bboxes_norm = [
            boxutils.dfbox_to_norm(box_row) for _, box_row in masks_df.iterrows()
        ]
        masks = [
            coco_mask_decode({"counts": row[1]["mask"], "size": [100, 100]})
            for row in masks_df.iterrows()
        ]
        masks = [
            resize_mask(
                mask, height=int(1080 * resize_factor), width=int(1920 * resize_factor)
            )
            for mask in masks
        ]

        colors = [get_masks_color(obj[1]) for obj in masks_df.iterrows()]
        labels = [get_masks_label(obj[1]) for obj in masks_df.iterrows()]
        detect2d.visualize_bboxes(
            ax,
            bboxes_norm,
            labels=labels,
            label_color="w",
            linewidth=2,
            color=colors,
        )
        for label, mask in zip(labels, masks):
            base_mask = mask[:, :, np.newaxis].astype(np.float)
            # show_mask = np.concatenate(
            #     [base_mask.repeat(3, 2), base_mask * alpha_mask], 2
            # )
            show_mask = np.concatenate(
                [base_mask, base_mask * 0.7, base_mask * 0.8, base_mask * alpha_mask], 2
            )
            ax.imshow(show_mask)


def get_masks_color(obj):
    return "c"


def get_masks_label(obj):
    return f"{obj.label}: {obj.score:.2f}"
