import torch
from kornia.geometry.camera import perspective

from epic.rendering import py3drendutils
from libyana.renderutils import catmesh


def add_hands_viz(ax, img, pred_hands, camintr, faces_per_pixel=2):
    # Create batched inputs
    camintr_th = torch.Tensor(camintr).unsqueeze(0).cuda()
    all_verts = [
        torch.Tensor(pred["verts"]).unsqueeze(0).cuda()
        for pred in pred_hands.values()
    ]
    all_faces = [
        torch.Tensor(pred["faces"].copy()).unsqueeze(0).cuda()
        for pred in pred_hands.values()
    ]
    verts, faces, _ = catmesh.batch_cat_meshes(all_verts, all_faces)
    # Convert vertices from weak perspective to camera
    unproj3d = perspective.unproject_points(
        verts[:, :, :2], verts[:, :, 2:] / 200 + 0.5, camintr_th
    )

    # Render
    res = py3drendutils.batch_render(
        unproj3d,
        faces,
        faces_per_pixel=faces_per_pixel,
        K=camintr_th,
        image_sizes=[(img.shape[1], img.shape[0])],
    )
    ax.imshow(res[0].detach().cpu()[:, :, :4], alpha=0.6)
