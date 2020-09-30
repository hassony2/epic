from collections import defaultdict
import os
from pathlib import Path

import cv2
from matplotlib import pyplot as plt
import numpy as np
from pytorch3d.io import load_obj as py3dload_obj
import torch
from tqdm import tqdm

from epic.rendering.py3drendutils import batch_render
from epic.lib3d import rotations
from libyana.visutils import vizmp
from libyana.conversions import npt
from libyana.metrics import iou as lyiou
from robust_loss_pytorch.adaptive import AdaptiveLossFunction

os.environ["FFMPEG_BINARY"] = "/sequoia/data3/yhasson/miniconda3/bin/ffmpeg"
import moviepy.editor as mpy


def distance_transform(mask, mask_size=5, pixel_scaling=200):
    mask_np = mask.cpu().detach().numpy().astype(np.uint8)
    dtf = cv2.distanceTransform(
        1 - mask_np, distanceType=cv2.DIST_L2, maskSize=mask_size
    )
    return 1 - dtf / pixel_scaling


def normalize_obj_verts(verts, radius=1):
    if verts.dim() != 2 or verts.shape[1] != 3:
        raise ValueError(f"Expected verts in format (vert_nb, 3), got {verts.shape}")

    # Center object
    verts = verts - verts.mean(0)
    assert torch.allclose(verts.mean(0), torch.zeros_like(verts.mean(0)), atol=1e-6)

    # Scale
    verts = radius * verts / verts.norm(2, -1).max()
    return verts


def fitobj2mask(
    masks,
    obj_paths,
    z_off=0.5,
    radius=0.1,
    faces_per_pixel=1,
    lr=0.01,
    loss_type="l2",
    iters=100,
    viz_step=1,
    save_folder="tmp/",
    viz_rows=4,
):
    # Initialize logging info
    opts = {
        "z_off": z_off,
        "loss_type": loss_type,
        "iters": iters,
        "radius": radius,
        "lr": lr,
        "obj_paths": obj_paths,
        "faces_per_pix": faces_per_pixel,
    }
    results = {"opts": opts}
    save_folder = Path(save_folder)
    metrics = defaultdict(list)

    batch_size = len(obj_paths)
    # Load normalized object
    batch_faces = []
    batch_verts = []
    for obj_path in obj_paths:
        verts_loc, faces_idx, _ = py3dload_obj(obj_path)
        faces = faces_idx.verts_idx
        batch_faces.append(faces.cuda())

        verts = normalize_obj_verts(verts_loc, radius).cuda()
        batch_verts.append(verts)
    batch_verts = torch.stack(batch_verts)
    batch_faces = torch.stack(batch_faces)

    # Dummy intrinsic camera
    height, width = masks[0].shape
    focal = min(masks[0].shape)
    camintr = (
        torch.Tensor([[focal, 0, width // 2], [0, focal, height // 2], [0, 0, 1]])
        .cuda()
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
    )

    adaptive_loss = AdaptiveLossFunction(
        num_dims=height * width, float_dtype=np.float32, device="cuda:0"
    )
    # Prepare rigid parameters
    rot_vec = torch.Tensor([[1, 0, 0, 0, 1, 0] for _ in range(batch_size)]).cuda()
    trans = torch.Tensor([[0, 0, z_off] for _ in range(batch_size)]).cuda()
    trans.requires_grad = True
    rot_vec.requires_grad = True
    optim_params = [rot_vec, trans]
    if "adapt" in loss_type:
        optim_params = optim_params + list(adaptive_loss.parameters())
    optimizer = torch.optim.Adam([rot_vec, trans], lr=lr)

    ref_masks = torch.stack(masks).cuda()
    # Prepare reference mask
    if "dtf" in loss_type:
        target_masks = torch.stack(
            [torch.Tensor(distance_transform(mask)) for mask in masks]
        ).cuda()
    else:
        target_masks = ref_masks

    col_nb = 5
    fig_res = 1.5
    # Aggregate images
    clip_data = []
    for iter_idx in tqdm(range(iters)):
        rot_mat = rotations.compute_rotation_matrix_from_ortho6d(rot_vec)
        optim_verts = batch_verts.bmm(rot_mat) + trans.unsqueeze(1)
        rendres = batch_render(
            optim_verts,
            batch_faces,
            K=camintr,
            image_sizes=[(width, height)],
            mode="silh",
            faces_per_pixel=faces_per_pixel,
        )
        optim_masks = rendres[:, :, :, -1]
        mask_diff = ref_masks - optim_masks
        mask_l2 = (mask_diff ** 2).mean()
        mask_l1 = mask_diff.abs().mean()
        mask_iou = lyiou.batch_mask_iou((optim_masks > 0), (ref_masks > 0)).mean()
        metrics["l1"].append(mask_l1.item())
        metrics["l2"].append(mask_l2.item())
        metrics["mask"].append(mask_iou.item())

        optim_mask_diff = target_masks - optim_masks
        if "l2" in loss_type:
            loss = (optim_mask_diff ** 2).mean()
        elif "l1" in loss_type:
            loss = optim_mask_diff.abs().mean()
        elif "adapt" in loss_type:
            loss = adaptive_loss.lossfun(optim_mask_diff.view(batch_size, -1)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter_idx % viz_step == 0:
            row_idxs = np.linspace(0, batch_size - 1, viz_rows).astype(np.int)
            row_nb = viz_rows
            fig, axes = plt.subplots(
                row_nb, col_nb, figsize=(int(col_nb * fig_res), int(row_nb * fig_res))
            )
            for row_idx in range(row_nb):
                show_idx = row_idxs[row_idx]
                ax = vizmp.get_axis(axes, row_idx, 0, row_nb=row_nb, col_nb=col_nb)
                ax.imshow(npt.numpify(optim_masks[show_idx]))
                ax.set_title("optim mask")
                ax = vizmp.get_axis(axes, row_idx, 1, row_nb=row_nb, col_nb=col_nb)
                ax.imshow(npt.numpify(ref_masks[show_idx]))
                ax.set_title("ref mask")
                ax = vizmp.get_axis(axes, row_idx, 2, row_nb=row_nb, col_nb=col_nb)
                ax.imshow(
                    npt.numpify(ref_masks[show_idx] - optim_masks[show_idx]),
                    vmin=-1,
                    vmax=1,
                )
                ax.set_title("ref masks diff")
                ax = vizmp.get_axis(axes, row_idx, 3, row_nb=row_nb, col_nb=col_nb)
                ax.imshow(npt.numpify(target_masks[show_idx]), vmin=-1, vmax=1)
                ax.set_title("target mask")
                ax = vizmp.get_axis(axes, row_idx, 4, row_nb=row_nb, col_nb=col_nb)
                ax.imshow(
                    npt.numpify(target_masks[show_idx] - optim_masks[show_idx]),
                    vmin=-1,
                    vmax=1,
                )
                ax.set_title("masks diff")
            viz_folder = save_folder / "viz"
            viz_folder.mkdir(parents=True, exist_ok=True)
            data = vizmp.fig2np(fig)
            clip_data.append(data)
            fig.savefig(viz_folder / f"{iter_idx:04d}.png")

    clip = mpy.ImageSequenceClip(clip_data, fps=4)
    clip.write_videofile(str(viz_folder / "out.mp4"))
    clip.write_videofile(str(viz_folder / "out.webm"))
    results["metrics"] = metrics
    return results
