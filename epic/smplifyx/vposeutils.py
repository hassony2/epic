import torch

from human_body_prior.tools.model_loader import load_vposer
import smplx
from human_body_prior.body_model.body_model import BodyModel


def get_vposer_mean_pose(vposer_dir="misc/vposer_dir"):
    pose_embedding = torch.zeros([1, 32])
    vposer, _ = load_vposer(vposer_dir, vp_model="snapshot")
    vposer.eval()

    mean_pose = (
        vposer.decode(pose_embedding, output_type="aa")
        .contiguous()
        .view(1, -1)
    )
    return mean_pose


def get_smplx(model_root="misc/models", vposer_dir="misc/vposer_dir"):
    model_params = dict(
        model_path=model_root,
        model_type="smplx",
        gender="neutral",
        create_body_pose=True,
        dtype=torch.float32,
        use_face=False,
    )
    bm_path = "assets/models/smplx/SMPLX_NEUTRAL.npz"
    model = BodyModel(bm_path=bm_path, batch_size=1).to("cuda")
    model = smplx.create(**model_params)

    # Initialize body parameters.
    betas = torch.zeros([1, 10], dtype=torch.float32)
    expression = torch.randn([1, 10], dtype=torch.float32)
    mean_pose = get_vposer_mean_pose(vposer_dir=vposer_dir)
    # Set them.
    model.expression.data.copy_(expression)
    model.betas.data.copy_(betas)
    model.body_pose.data.copy_(mean_pose)
    return model


def get_vposer_pose(smpl_model, vposer, pose_embedding):
    batch_size = pose_embedding.shape[0]
    body_pose = vposer.decode(pose_embedding, output_type="aa").view(
        batch_size, -1
    )
    # vposer_mean_pose = smpl_model.body_pose.repeat(batch_size, 1)
    # body_pose = body_pose - vposer_mean_pose
    return body_pose
