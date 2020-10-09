from collections import defaultdict
from pathlib import Path
import pickle
import warnings

import torch
from tqdm import tqdm
from epic.smplifyx import optim_factory, camera
from epic.egofit import egoviz


def fit_human(
    egohuman,
    fit_data,
    camintr=None,
    rot=None,
    iters=100,
    lr=0.1,
    optimizer="adam",
    save_root="tmp",
    optim_shape=False,
    viz_step=10,
):
    save_folder = Path(save_root) / f"opt{optimizer}_lr{lr:.4f}_it{iters:04d}"
    save_folder.mkdir(exist_ok=True, parents=True)
    fit_info = egohuman.preprocess_supervision(fit_data)
    gt_proj_verts = fit_info["verts"][:, :]
    vert_flags = fit_info["verts_confs"]
    batch_size = len(fit_info["imgs"])
    smpl_params = egohuman.get_optim_params(
        optim_shape=optim_shape, batch_size=batch_size
    )
    pose_embedding = smpl_params["pose_embedding"]
    left_hand_pose = smpl_params["left_hand_pose"]
    right_hand_pose = smpl_params["right_hand_pose"]

    fx, fy = camintr[0, 0], camintr[1, 1]
    center = camintr[:2, 2]
    cam = camera.create_camera(
        focal_length_x=fx,
        focal_length_y=fy,
        center=torch.Tensor(center).unsqueeze(0),
        rotation=rot.unsqueeze(0),
    )

    # Initialize optimizer
    (body_optimizer, body_create_graph,) = optim_factory.create_optimizer(
        list(smpl_params.values()), optim_type=optimizer, lr=lr
    )
    losses = defaultdict(list)
    for iter_idx in tqdm(range(iters)):
        body_model_output = egohuman.smpl_forward_vposer(
            pose_embedding,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
        )
        pred_verts = body_model_output.vertices
        pred_proj_verts = cam(pred_verts)
        vert_diffs = (pred_proj_verts - gt_proj_verts[:, :, :2]) / 100
        vert_losses = vert_flags.unsqueeze(-1) * vert_diffs
        # vert_losses_sq = (vert_losses).abs()
        # print((vert_losses != vert_losses).sum())
        # print(vert_losses.min(), vert_losses.max())
        vert_losses_sq = vert_losses.pow(2)
        loss_vert = (vert_losses_sq).sum() / vert_flags.sum()
        loss_embed = (pose_embedding ** 2).sum()
        loss = loss_vert + 0.000001 * loss_embed
        warnings.warn("epic/smplifyx/fitting.py line 436 only 2D supervision")
        body_optimizer.zero_grad()
        loss.backward()
        body_optimizer.step()
        if iter_idx % viz_step == 0:
            egoviz.ego_viz(
                pred_verts,
                pred_proj_verts,
                gt_proj_verts,
                vert_flags=vert_flags,
                cam=cam,
                faces=egohuman.armfaces,
                imgs=fit_info["imgs"],
                step_idx=iter_idx,
                save_folder=save_folder / "viz",
            )
        losses["verts"].append(loss_vert.item())
        losses["vposer"].append(loss_embed.item())
        losses["loss"].append(loss.item())
    res = {
        "losses": losses,
        "pose_embedding": pose_embedding,
        "left_hand_pose": left_hand_pose,
        "right_hand_pose": right_hand_pose,
    }
    with (save_folder / "res.pkl").open("rb") as p_f:
        pickle.dump(res, p_f)
    return res
