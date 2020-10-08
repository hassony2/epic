import pickle

import numpy as np
import torch

from epic.rendering.py3drendutils import batch_render

# from handobjectdatasets.viz2d import visualize_joints_2d_cv2

LH_IDXS = list(range(25, 40)) + [20]
RH_IDXS = list(range(40, 55)) + [21]
LFOREARM_IDXS = [18]
RFOREARM_IDXS = [19]
LARM_IDXS = [16]
RARM_IDXS = [17]


def filter_faces(faces, vertex_ranges=None):
    all_faces = []
    for vertex_range in vertex_ranges:
        select_faces = faces[
            :,
            (
                ((faces > vertex_range[0]) * (faces < vertex_range[1])).sum(-1)
                >= 1
            )[0],
        ]
        all_faces.append(select_faces)
    filtered_faces = torch.cat(all_faces, dim=1)
    return filtered_faces


def filter_parts(
    faces, part_pkl_path="misc/models/smplx_parts_segm.pkl", part_idxs=None
):
    if part_idxs is None:
        # part_idxs = LH_IDXS + RH_IDXS + LFOREARM_IDXS +
        # RFOREARM_IDXS + LARM_IDXS + RARM_IDXS
        part_idxs = LH_IDXS + RH_IDXS + LFOREARM_IDXS + RFOREARM_IDXS
    with open(part_pkl_path, "rb") as p_f:
        parts_segm = pickle.load(p_f, encoding="latin1")
    all_faces = []
    for part_idx in part_idxs:
        part_faces = faces[parts_segm["segm"] == part_idx]
        all_faces.append(torch.Tensor(part_faces.astype(np.int32)))
    filtered_faces = torch.cat(all_faces, dim=0)
    return filtered_faces


def show_fits(
    axes,
    frame,
    joints2d=None,
    vertices=None,
    arm_faces=None,
    cam_rot=None,
    cam_trans=None,
    camintr_th=None,
    img_shape=None,
    offset=(0, 0, 0.8),
):
    opp_links = [
        [0, 1, 8, 9, 10, 11, 22, 23],
        [4, 3, 2, 1, 5, 6, 7],
        [8, 12, 13, 14, 19, 20],
    ]
    hand_links = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],
    ]
    for hand_off in [25, 46]:
        for finger_link in hand_links:
            off_finger_link = [hand_off + joint for joint in finger_link]
            opp_links += [off_finger_link]
    ax = axes[0, 0]
    ax.imshow(frame[:, :, ::-1])
    ax.axis("off")

    rendered = batch_render(
        vertices.cuda(),
        arm_faces.int(),
        img_shape,
        camintrs=camintr_th,
        cam_rot=cam_rot,
        cam_trans=cam_trans,
    )
    ax = axes[0, 1]
    ax.axis("off")
    ax.imshow(frame[:, :, ::-1])
    ax.imshow(rendered[0].detach().cpu(), alpha=0.9)

    # Offset
    rendered_mirr = batch_render(
        (vertices + vertices.new(offset)).cuda(),
        arm_faces.int(),
        img_shape,
        camintrs=camintr_th,
        cam_rot=cam_rot,
        cam_trans=cam_trans,
    )
    ax = axes[1, 0]
    ax.axis("off")
    ax.imshow(
        torch.zeros_like(rendered_mirr[0, :, :, :3].detach().cpu()), alpha=0.5
    )
    ax.imshow(rendered_mirr[0].detach().cpu().numpy()[:, ::-1], alpha=0.5)
    # Offset mirrored
    ax = axes[1, 1]
    ax.axis("off")
    ax.imshow(
        torch.zeros_like(rendered[0, :, :, :3].detach().cpu()), alpha=0.5
    )
    ax.imshow(rendered[0].detach().cpu().numpy(), alpha=0.5)
