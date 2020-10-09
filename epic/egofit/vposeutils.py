import torch

from human_body_prior.tools.model_loader import load_vposer
import smplx


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


def get_smplx(model_root="misc/models"):
    model_params = dict(
        model_path=model_root,
        model_type="smplx",
        gender="neutral",
        create_body_pose=True,
        dtype=torch.float32,
        use_face=False,
    )

    model = smplx.create(**model_params)
    print(model)

    # Initialize body parameters.
    betas = torch.zeros([1, 10], dtype=torch.float32)
    expression = torch.randn([1, 10], dtype=torch.float32)
    mean_pose = get_vposer_mean_pose()
    # Set them.
    model.expression.data.copy_(expression)
    model.betas.data.copy_(betas)
    model.body_pose.data.copy_(mean_pose)
    return model
