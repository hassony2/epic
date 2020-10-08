# Loading SMPLx Body Model
import torch
from matplotlib import pyplot as plt

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.model_loader import load_vposer
import smplx

torch.seed()
# torch.autograd.set_detect_anomaly(True)
mode = "smplx"
if mode == "smplx":
    model_params = dict(
        model_path="assets/models",
        model_type="smplx",
        gender="female",
        # create_body_pose=True,
        dtype=torch.float32,
        use_face=False,
    )
    model = smplx.create(**model_params)
    model.cuda()
else:
    bm_path = "assets/models/smplx/SMPLX_FEMALE.npz"
    bm = BodyModel(bm_path=bm_path, batch_size=1).to("cuda")
    bm.cuda()
vp, ps = load_vposer("assets/vposer")
vp = vp.to("cuda")
vp.eval()

# Sample a 32 dimentional vector from a Normal distribution
poZ_body_sample = torch.zeros(1, 32).cuda()
pose_body = vp.decode(poZ_body_sample, output_type="aa").view(-1, 63)


pose_embedding = torch.zeros(
    [1, 32], requires_grad=True, device=poZ_body_sample.device
)
pose_embedding = torch.rand(
    [1, 32], requires_grad=True, device=poZ_body_sample.device
)

optimizer = torch.optim.Adam([pose_embedding], lr=0.1)
target_locs = torch.Tensor([[0.3, 0, 0.5], [-0.3, 0, 0.5]]).cuda()

wrist_idxs = list(range(20, 22))
for iter_idx in range(100):
    pose_body_off = vp.decode(pose_embedding, output_type="aa").view(-1, 63)
    if mode == "smplx":
        # outputs = model.forward(body_pose=pose_body_off - pose_body)
        # outputs = model.forward(body_pose=pose_body_off - pose_body)
        outputs = model.forward(body_pose=pose_body_off)
        verts = outputs.vertices
        joints = outputs.joints
    else:
        # out = bm.forward(pose_body=pose_body_off - pose_body, return_vertices=True)
        out = bm.forward(pose_body=pose_body_off, return_vertices=True)
        verts = out.v
        joints = out.Jtr
    wrist_joints = joints[:, wrist_idxs]
    diffs = wrist_joints - target_locs
    loss = (diffs ** 2).sum(-1).mean() + 0.001 * (pose_embedding ** 2).sum()
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    # if (pose_embedding != pose_embedding).sum():
    #     import pdb; pdb.set_trace()

    pts = verts.cpu().detach().numpy()
    wrists = joints[0, wrist_idxs].cpu().detach().numpy()
    targ = target_locs.cpu().detach().numpy()
    plt.clf()
    fig, axes = plt.subplots(1, 3)
    ax = axes[0]
    ax.scatter(pts[0, :, 0], pts[0, :, 1], s=1)
    ax.scatter(wrists[:, 0], wrists[:, 1], s=10)
    ax.scatter(targ[0][0], targ[0][1], s=10)
    ax.scatter(targ[1][0], targ[1][1], s=10)

    ax = axes[1]
    ax.scatter(pts[0, :, 0], pts[0, :, 2], s=1)
    ax.scatter(wrists[:, 0], wrists[:, 2], s=10)
    ax.scatter(targ[0][0], targ[0][2], s=10)
    ax.scatter(targ[1][0], targ[1][2], s=10)
    ax.axis("equal")

    ax = axes[2]
    ax.scatter(pts[0, :, 1], pts[0, :, 2], s=1)
    ax.scatter(wrists[:, 1], wrists[:, 2], s=10)
    ax.scatter(targ[0][1], targ[0][2], s=10)
    ax.scatter(targ[1][1], targ[1][2], s=10)
    ax.axis("equal")
    fig.savefig(f"tmp_{iter_idx:04d}.png")
print(pose_embedding)
