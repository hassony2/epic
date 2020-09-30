from collections import defaultdict
from pathlib import Path
from matplotlib import pyplot as plt
from pytorch3d.io import load_obj as py3dload_obj
import torch
from tqdm import tqdm

from epic.rendering.py3drendutils import batch_render
from epic.lib3d import rotations
from epic.viz import vizutils


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
    mask,
    obj_path,
    z_off=0.5,
    radius=0.1,
    lr=0.01,
    loss_type="l2",
    iters=100,
    viz_step=1,
    save_folder="tmp/",
):
    save_folder = Path(save_folder)
    # Load normalized object
    verts_loc, faces_idx, _ = py3dload_obj(obj_path)
    faces = faces_idx.verts_idx.unsqueeze(0).cuda()
    verts = normalize_obj_verts(verts_loc, radius).unsqueeze(0).cuda()

    # Dummy intrinsic camera
    height, width = mask.shape
    focal = min(mask.shape)
    camintr = (
        torch.Tensor([[focal, 0, width // 2], [0, focal, height // 2], [0, 0, 1]])
        .cuda()
        .unsqueeze(0)
    )

    # Prepare rigid parameters
    rot_vec = torch.Tensor([[1, 0, 0, 0, 1, 0]]).cuda()
    trans = torch.Tensor([[0, 0, z_off]]).cuda()
    trans.requires_grad = True
    rot_vec.requires_grad = True
    optimizer = torch.optim.Adam([rot_vec, trans], lr=lr)

    # Prepare reference mask
    mask = mask.cuda().unsqueeze(0)

    col_nb = 3
    row_nb = len(verts)
    fig_res = 4
    metrics = defaultdict(list)
    for iter_idx in tqdm(range(iters)):
        rot_mat = rotations.compute_rotation_matrix_from_ortho6d(rot_vec)
        optim_verts = verts.bmm(rot_mat) + trans
        res = batch_render(optim_verts, faces, K=camintr, image_sizes=[(width, height)])
        optim_mask = res[:, :, :, -1]
        mask_diff = mask - optim_mask
        mask_l2 = (mask_diff ** 2).mean()
        mask_l1 = mask_diff.abs().mean()
        if loss_type == "l2":
            loss = mask_l2
        elif loss_type == "l1":
            loss = mask_l1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter_idx % viz_step == 0:
            fig, axes = plt.subplots(
                1, col_nb, figsize=(int(col_nb * fig_res), int(row_nb * fig_res))
            )
            for row_idx in range(row_nb):
                ax = vizutils.get_axis(axes, row_idx, 0, row_nb=row_nb, col_nb=col_nb)
                ax.imshow(vizutils.numpify(optim_mask[row_idx]))
                ax.set_title("optim mask")
                ax = vizutils.get_axis(axes, row_idx, 1, row_nb=row_nb, col_nb=col_nb)
                ax.imshow(vizutils.numpify(mask[row_idx]))
                ax.set_title("ref mask")
                ax = vizutils.get_axis(axes, row_idx, 2, row_nb=row_nb, col_nb=col_nb)
                ax.imshow(vizutils.numpify(mask[row_idx] - optim_mask[row_idx]))
                ax.set_title("masks diff")
            viz_folder = save_folder / "viz"
            viz_folder.mkdir(parents=True, exist_ok=True)
            fig.savefig(viz_folder / f"{iter_idx:04d}.png")

    plt.imshow(res[0, :, :, 0].cpu())
    plt.savefig("tmp.png")
    plt.imshow(res[0, :, :, -1].cpu())
    plt.savefig("mask.png")
    import pdb

    pdb.set_trace()
