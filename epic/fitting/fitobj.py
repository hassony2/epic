from collections import defaultdict
import os
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from pytorch3d.io import load_obj as py3dload_obj
from scipy.stats import special_ortho_group 
import torch
from tqdm import tqdm

from epic.rendering.py3drendutils import batch_render
from epic.lib3d import rotations, ops3d, camutils, normalize
from epic.lib2d import cropping, boxutils, dtf
from libyana.visutils import vizmp
from libyana.conversions import npt
from libyana.metrics import iou as lyiou
from robust_loss_pytorch.adaptive import AdaptiveLossFunction

os.environ["FFMPEG_BINARY"] = "/sequoia/data3/yhasson/miniconda3/bin/ffmpeg"
import moviepy.editor as mpy


def repeatdim(tens, repeat_nb, undim=0):
    """
    Creates copies of tensor 
    """
    if repeat_nb == 1:
        flat_tens = tens
    else:
        assert undim in [0, 1], f"undim {undim} not in [0,1]"
        or_shape = tens.shape
        repeat_vals = [1 for _ in range(tens.dim() + 1)]
        repeat_vals[undim] = repeat_nb
        rep_tens = tens.unsqueeze(undim).repeat(*repeat_vals)
        flat_tens = rep_tens.view(or_shape[0] * repeat_nb, *or_shape[1:])
    return flat_tens


def fitobj2mask(
    masks,
    bboxes,
    obj_paths,
    z_off=0.5,
    radius=0.1,
    faces_per_pixel=1,
    lr=0.01,
    loss_type="l2",
    iters=100,
    viz_step=1,
    save_folder="tmp/",
    viz_rows=12,
    crop_box=True,
    crop_size=(200, 200),
    rot_nb=1,
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

        verts = normalize.normalize_verts(verts_loc, radius).cuda()
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

    if crop_box:
        adaptive_loss = AdaptiveLossFunction(
            num_dims=crop_size[0] * crop_size[1],
            float_dtype=np.float32,
            device="cuda:0",
        )
    else:
        adaptive_loss = AdaptiveLossFunction(
            num_dims=height * width, float_dtype=np.float32, device="cuda:0"
        )
    # Prepare rigid parameters
    if rot_nb > 1:
        rot_mats = [special_ortho_group.rvs(3) for _ in range(rot_nb)]
        rot_vecs = torch.Tensor([np.linalg.svd(rot_mat)[0][:2].reshape(-1) for rot_mat in rot_mats])
        rot_vec = rot_vecs.repeat(batch_size, 1) .cuda()
        # Ordering b1 rot1, b1 rot2, ..., b2 rot1, ...
    else:
        rot_vec = torch.Tensor([[1, 0, 0, 0, 1, 0] for _ in range(batch_size)]).cuda()

    bboxes = torch.stack(bboxes)
    trans = ops3d.trans_init_from_boxes(bboxes, camintr, (z_off, z_off)).cuda()
    # Repeat to match rots
    trans = repeatdim(trans, rot_nb, 1)
    bboxes = boxutils.pad(bboxes)
    if crop_box:
        camintr_crop = camutils.get_K_crop_resize(camintr, bboxes, crop_size)
    camintr_crop = repeatdim(camintr_crop, rot_nb, 1)

    trans.requires_grad = True
    rot_vec.requires_grad = True
    optim_params = [rot_vec, trans]
    if "adapt" in loss_type:
        optim_params = optim_params + list(adaptive_loss.parameters())
    optimizer = torch.optim.Adam([rot_vec, trans], lr=lr)

    ref_masks = torch.stack(masks).cuda()
    if crop_box:
        ref_masks = cropping.crops(ref_masks.float(), bboxes, crop_size)[:, 0]

    # Prepare reference mask
    if "dtf" in loss_type:
        target_masks = torch.stack(
            [torch.Tensor(dtf.distance_transform(mask)) for mask in ref_masks]
        ).cuda()
    else:
        target_masks = ref_masks
    ref_masks = repeatdim(ref_masks, rot_nb, 1)
    target_masks = repeatdim(target_masks, rot_nb, 1)
    batch_verts = repeatdim(batch_verts, rot_nb, 1)
    batch_faces = repeatdim(batch_faces, rot_nb, 1)

    col_nb = 5
    fig_res = 1.5
    # Aggregate images
    clip_data = []
    for iter_idx in tqdm(range(iters)):
        rot_mat = rotations.compute_rotation_matrix_from_ortho6d(rot_vec)
        optim_verts = batch_verts.bmm(rot_mat) + trans.unsqueeze(1)
        if crop_box:
            rendres = batch_render(
                optim_verts,
                batch_faces,
                K=camintr_crop,
                image_sizes=[(crop_size[1], crop_size[0])],
                mode="silh",
                faces_per_pixel=faces_per_pixel,
            )
        else:
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
            loss = adaptive_loss.lossfun(optim_mask_diff.view(rot_nb * batch_size, -1)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter_idx % viz_step == 0:
            row_idxs = np.linspace(0, batch_size * rot_nb - 1, viz_rows).astype(np.int)
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
