import os
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import torch

from libyana.visutils import vizmp
from epic.rendering.py3drendutils import batch_render


def ego_viz(
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
    print(save_path)
    print(f"Saved to {save_path}")
