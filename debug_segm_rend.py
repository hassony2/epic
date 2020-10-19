import argparse
import cv2
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from libyana.exputils import argutils
from libyana.visutils import vizmp
from libyana.conversions import npt
from libyana.renderutils import catmesh

from epic.egofit import debobj
from epic.egofit import scene
from epic.egofit import egoviz
from epic.rendering.py3drendutils import batch_render
from epic.smplifyx import optim_factory


matplotlib.use("agg")


parser = argparse.ArgumentParser()
parser.add_argument("--iter_nb", default=200, type=int)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--batch_size", default=3)
parser.add_argument("--viz_step", default=10, type=int)
parser.add_argument("--faces_per_pixel", default=10, type=int)
parser.add_argument("--block_obj_scale", action="store_true")
parser.add_argument("--loss_type", default="l1", choices=["l1", "l2", "adapt"])
args = parser.parse_args()
argutils.print_args(args)

obj_nb = 3
z_offs = [0.8, 0.8, 0.8]
y_offs = [0.1, -0.1, 0]
x_offs = [0, 0, 0.5]
radiuses = [0.5, 0.5, 0.5]
objects = []
for obj_idx in range(obj_nb):
    obj = debobj.IcoSphere(
        z_off=z_offs[obj_idx],
        y_off=y_offs[obj_idx],
        x_off=x_offs[obj_idx],
        batch_size=args.batch_size,
        radius=radiuses[obj_idx],
    )
    objects.append(obj.cuda())

camintr = (
    torch.Tensor(np.array([[100, 0, 100], [0, 100, 150], [0, 0, 1]]))
    .cuda()
    .unsqueeze(0)
    .repeat(args.batch_size, 1, 1)
)
device = camintr.device

img = np.zeros((200, 300, obj_nb))

img = cv2.rectangle(img, (50, 50), (150, 150), (1, 0, 0), -1)
img = cv2.rectangle(img, (1, 1), (100, 100), (0, 1, 0), -1)
img = cv2.rectangle(img, (200, 100), (250, 150), (0, 0, 1), -1)
img_th = (
    torch.tensor(img).unsqueeze(0).repeat(args.batch_size, 1, 1, 1).to(device)
)
# Add alpha channel
img_th = torch.cat([img_th, img_th.sum(-1, keepdim=True).clamp(0, 1)], -1)

# Initialize objects
params = []
for obj in objects:
    obj_params = obj.get_params(optim_scale=not args.block_obj_scale)
    params.extend(obj_params)

optimizer, _ = optim_factory.create_optimizer(
    params, optim_type="adam", lr=args.lr
)

for iter_idx in tqdm(range(args.iter_nb)):
    obj_infos = [obj.forward() for obj in objects]
    verts = [obj_info["verts"] for obj_info in obj_infos]
    faces = [obj_info["faces"] for obj_info in obj_infos]
    colors = scene.get_segm_colors(verts, faces)
    all_verts, all_faces, all_colors = catmesh.batch_cat_meshes(
        verts, faces, colors
    )
    rendres = batch_render(
        all_verts,
        all_faces,
        K=camintr,
        image_sizes=[(300, 200)],
        shading="soft",
        mode="facecolor",
        # mode="rgb",
        face_colors=colors,
        color=(1, 0, 0),
        faces_per_pixel=args.faces_per_pixel,
    )
    row_nb = args.batch_size
    col_nb = 3
    diffs = (rendres - img_th[:, :, :, :])[:, :, :, :3]
    if args.loss_type == "l1":
        loss = diffs.abs().mean()
    if args.loss_type == "l2":
        loss = (diffs ** 2).sum(-1).mean()
    # loss = (rendres - img_th).abs().mean()
    optimizer.zero_grad()
    loss.backward()
    print(loss)
    optimizer.step()
    if iter_idx % args.viz_step == 0:
        fig, axes = plt.subplots(row_nb, col_nb)
        for row_idx in range(row_nb):
            ax = vizmp.get_axis(
                axes, row_idx=row_idx, col_idx=0, row_nb=row_nb, col_nb=col_nb
            )
            ax.imshow(egoviz.imagify(rendres[row_idx], normalize_colors=False))
            ax = vizmp.get_axis(
                axes, row_idx=row_idx, col_idx=1, row_nb=row_nb, col_nb=col_nb
            )
            ax.imshow(
                egoviz.imagify(img_th[row_idx][:, :], normalize_colors=False)
            )
            ax = vizmp.get_axis(
                axes, row_idx=row_idx, col_idx=2, row_nb=row_nb, col_nb=col_nb
            )
            ax.imshow(npt.numpify(diffs[row_idx]))
        fig.savefig(f"tmp_{iter_idx:04d}.png", bbox_inches="tight")
