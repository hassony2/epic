import os
from matplotlib import pyplot as plt
from matplotlib import cm, ticker
import numpy as np
import torch

from libyana.conversions import npt
from libyana.visutils import vizmp
from epic.rendering.py3drendutils import batch_render


def add_hands(ax, sample_data):
    for side in ["left", "right"]:
        if side in sample_data["hands"]:
            hand_data = sample_data["hands"][side]
            hand_verts = hand_data["verts"]
            # vert_nb = hand_verts.shape[0]
            # colors = cm.rainbow(np.linspace(0, 1, vert_nb))
            if side == "right":
                colors = "c"
            elif side == "left":
                colors = "m"
            vert_pt_step = 10
            ax.scatter(
                hand_verts[::vert_pt_step, 0],
                hand_verts[::vert_pt_step, 1],
                s=0.5,
                alpha=0.2,
                c=colors,
            )


def make_alpha(img, bg_val=1):
    img = npt.numpify(img)

    # Extend single-channel image to 3 channels
    if img.ndim == 2:
        img = img[:, :, None]
    if img.shape[2] == 1:
        img = img.repeat(3, 2)
    # Alpha as locations where image is != from background color
    mask = (img[:, :, :3].sum(-1) != (bg_val * 3)).astype(img.dtype)
    img = np.concatenate([img, mask[:, :, None]], 2)
    return img


def ego_viz(
    data,
    supervision,
    scene_outputs,
    fig_res=2,
    step_idx=0,
    save_folder="tmp",
    sample_nb=4,
):
    scene_rend = npt.numpify(scene_outputs["scene_rend"])
    viz_rends = [npt.numpify(rend) for rend in scene_outputs["scene_viz_rend"]]
    ref_hand_rends = supervision["ref_hand_rends"]
    col_nb = 3 + len(viz_rends)
    fig, axes = plt.subplots(
        sample_nb,
        col_nb,
        figsize=(int(3 / 2 * col_nb * fig_res), int(sample_nb * fig_res)),
    )
    scene_size = len(supervision["imgs"])
    sample_idxs = np.linspace(0, scene_size - 1, sample_nb).astype(np.int)

    for row_idx, sample_idx in enumerate(sample_idxs):
        img = supervision["imgs"][sample_idx][:, :, ::-1]
        sample_data = data[sample_idx]
        # Column 1: image and supervision
        ax = vizmp.get_axis(
            axes, row_idx=row_idx, col_idx=0, row_nb=sample_nb, col_nb=col_nb
        )
        ax.imshow(img)
        ax.axis("off")
        add_hands(ax, sample_data)

        # Column 2: Rendered prediction on top of GT hands
        ax = vizmp.get_axis(
            axes, row_idx=row_idx, col_idx=1, row_nb=sample_nb, col_nb=col_nb
        )
        ax.imshow(ref_hand_rends[sample_idx])
        ax.imshow(make_alpha(scene_rend[sample_idx][:, :, :3]), alpha=0.8)
        ax.axis("off")

        # Column 3: Rendered prediction
        ax = vizmp.get_axis(
            axes, row_idx=row_idx, col_idx=2, row_nb=sample_nb, col_nb=col_nb
        )
        ax.imshow(img)
        ax.imshow(make_alpha(scene_rend[sample_idx][:, :, :3]), alpha=0.8)
        # Column 3: Rendered prediction
        ax.axis("off")
        for view_idx, viz_rend in enumerate(viz_rends):
            ax = vizmp.get_axis(
                axes,
                row_idx=row_idx,
                col_idx=3 + view_idx,
                row_nb=sample_nb,
                col_nb=col_nb,
            )
            ax.imshow(viz_rend[sample_idx, :, :, :3])
            ax.axis("off")

    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"tmp_{step_idx:04d}.png")
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.1, wspace=0)
    fig.gca().xaxis.set_major_locator(ticker.NullLocator())
    fig.gca().yaxis.set_major_locator(ticker.NullLocator())
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved to {save_path}")
    return save_path


def ego_viz_old(
    pred_verts,
    pred_proj_verts,
    gt_proj_verts,
    vert_flags,
    imgs=None,
    fig_res=2,
    cam=None,
    faces=None,
    step_idx=0,
    save_folder="tmp",
):
    # Render predicted human
    render_verts = pred_verts.cuda()
    batch_size = len(render_verts)
    faces_th = (
        faces.unsqueeze(0).repeat(batch_size, 1, 1).to(render_verts.device)
    )
    camintr = (
        cam.get_camintr()
        .to(render_verts.device)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
    )
    rot = (
        cam.get_camrot()
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
        .to(render_verts.device)
    )
    img_size = (imgs[0].shape[1], imgs[0].shape[0])
    with torch.no_grad():
        rends = batch_render(
            render_verts,
            faces_th,
            K=camintr,
            rot=rot,
            image_sizes=[img_size for _ in range(batch_size)],
        )

    show_pred_verts = pred_verts.cpu().detach().numpy()

    row_nb = len(pred_verts)
    col_nb = 4
    fig, axes = plt.subplots(
        row_nb, col_nb, figsize=(int(col_nb * fig_res), int(row_nb * fig_res))
    )
    for row_idx in range(row_nb):
        ax = vizmp.get_axis(
            axes, row_idx=row_idx, col_idx=0, row_nb=row_nb, col_nb=col_nb
        )
        super_proj_verts = (
            gt_proj_verts[row_idx][(vert_flags[row_idx] > 0)]
            .cpu()
            .detach()
            .numpy()
        )
        super_pred_proj_verts = (
            pred_proj_verts[row_idx][(vert_flags[row_idx] > 0)]
            .cpu()
            .detach()
            .numpy()
        )
        if imgs is not None:
            ax.imshow(imgs[row_idx])
        point_nb = super_pred_proj_verts.shape[0]
        colors = cm.rainbow(np.linspace(0, 1, point_nb))
        ax.scatter(
            super_proj_verts[:, 0],
            super_proj_verts[:, 1],
            s=0.5,
            alpha=0.2,
            c="k",
        )
        ax.scatter(
            super_pred_proj_verts[:, 0],
            super_pred_proj_verts[:, 1],
            s=0.5,
            alpha=0.2,
            c=colors,
        )
        ax.axis("equal")

        row_pred_verts = show_pred_verts[row_idx]
        ax = vizmp.get_axis(
            axes, row_idx=row_idx, col_idx=1, row_nb=row_nb, col_nb=col_nb
        )
        ax.scatter(row_pred_verts[:, 0], row_pred_verts[:, 2], s=1)
        ax.axis("equal")

        ax = vizmp.get_axis(
            axes, row_idx=row_idx, col_idx=2, row_nb=row_nb, col_nb=col_nb
        )
        ax.scatter(row_pred_verts[:, 1], row_pred_verts[:, 2], s=1)
        ax.axis("equal")

        ax = vizmp.get_axis(
            axes, row_idx=row_idx, col_idx=3, row_nb=row_nb, col_nb=col_nb
        )
        ax.imshow(imgs[row_idx])
        ax.imshow(rends[row_idx].sum(-1).cpu().numpy(), alpha=0.5)

    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"tmp_{step_idx:04d}.png")
    fig.savefig(save_path)
    print(f"Saved to {save_path}")
