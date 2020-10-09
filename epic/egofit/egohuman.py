import pickle

import numpy as np
import torch
from human_body_prior.tools.model_loader import load_vposer
from epic.smplifyx import (
    prior,
    vposeutils,
    smplvis,
    smplutils,
)
from epic.io.tarutils import TarReader


class EgoHuman:
    def __init__(
        self,
        debug=True,
        hand_pca_nb=6,
        head_center_idx=8949,
        mano_corresp_path="assets/models/MANO_SMPLX_vertex_ids.pkl",
        smpl_root="assets/models",
        vposer_dir="assets/vposer",
        vposer_dim=32,
        data_weight=1000 / 256,
        parts_path="assets/models/smplx/smplx_parts_segm.pkl",
    ):
        self.debug = debug
        self.hand_pca_nb = hand_pca_nb
        self.head_center_idx = head_center_idx
        self.smplx_vertex_nb = 10475
        self.vposer_dim = vposer_dim

        # Initialize SMPL-X model
        self.smpl_model = vposeutils.get_smplx(
            model_root=smpl_root, vposer_dir=vposer_dir
        )
        self.smpl_f = self.smpl_model.faces
        with open(mano_corresp_path, "rb") as p_f:
            self.mano_corresp = pickle.load(p_f)
        # Get vposer
        self.vposer = load_vposer(vposer_dir, vp_model="snapshot")[0]
        self.vposer.eval()
        self.tareader = TarReader()

        # Translate human so that head is at camera level
        self.set_head2cam_trans()

        self.armfaces = smplvis.filter_parts(self.smpl_f, parts_path)

        # Initialize priors
        self.body_pose_prior = prior.create_prior(prior_type="l2")
        self.hand_pca_nb = hand_pca_nb
        self.left_hand_prior = prior.create_prior(
            prior_type="l2",
            use_left_hand=True,
            num_gaussians=hand_pca_nb,
        )
        self.right_hand_prior = prior.create_prior(
            prior_type="l2",
            use_right_hand=True,
            num_gaussians=hand_pca_nb,
        )
        self.angle_prior = prior.create_prior(prior_type="angle")
        self.shape_prior = prior.create_prior(prior_type="l2")

    def set_head2cam_trans(self):
        pose_embedding = torch.zeros([1, self.vposer_dim])
        vpose = vposeutils.get_vposer_pose(
            self.smpl_model, self.vposer, pose_embedding
        )
        smplx_verts = self.smpl_model(
            betas=self.smpl_model.betas, body_pose=vpose, return_verts=True
        ).vertices
        head_loc = smplx_verts[:, self.head_center_idx]
        self.smpl_model.transl.data[0] = -(
            head_loc + head_loc.new([0, 0, 0.1])
        )

        vpose = vposeutils.get_vposer_pose(
            self.smpl_model, self.vposer, pose_embedding
        )
        smplx_verts = self.smpl_model(
            betas=self.smpl_model.betas, body_pose=vpose, return_verts=True
        ).vertices

    def smpl_forward(
        self, body_pose=None, right_hand_pose=None, left_hand_pose=None
    ):
        smpl_params = smplutils.prepare_params(
            self.smpl_model, body_pose.shape[0]
        )
        if right_hand_pose is not None:
            smpl_params["right_hand_pose"] = right_hand_pose
        if left_hand_pose is not None:
            smpl_params["left_hand_pose"] = left_hand_pose
        if body_pose is not None:
            smpl_params["body_pose"] = left_hand_pose
        body_model_output = self.smpl_model(
            body_pose=body_pose,
            expression=smpl_params["expression"],
            jaw_pose=smpl_params["jaw_pose"],
            global_orient=smpl_params["global_orient"],
            transl=smpl_params["transl"],
            leye_pose=smpl_params["leye_pose"],
            reye_pose=smpl_params["reye_pose"],
            left_hand_pose=smpl_params["left_hand_pose"],
            right_hand_pose=smpl_params["right_hand_pose"],
            return_verts=True,
            return_full_pose=True,
        )
        return body_model_output

    def smpl_forward_vposer(
        self, pose_embedding, right_hand_pose=None, left_hand_pose=None
    ):
        body_pose = vposeutils.get_vposer_pose(
            self.smpl_model, self.vposer, pose_embedding
        )
        body_model_output = self.smpl_forward(
            body_pose=body_pose,
            right_hand_pose=right_hand_pose,
            left_hand_pose=left_hand_pose,
        )
        return body_model_output

    def preprocess_supervision(self, fit_infos):
        sample_masks = []
        sample_verts = []
        sample_confs = []
        sample_imgs = []
        for fit_info in fit_infos:
            img = self.tareader.read_tar_frame(fit_info["img_path"])
            # img = cv2.imread(fit_info["img_path"])
            hand_infos = fit_info["hands"]
            human_verts = np.zeros((self.smplx_vertex_nb, 3))
            verts_confs = np.zeros((self.smplx_vertex_nb,))
            # Get hand vertex refernces poses
            for side in ["left", "right"]:
                if side in hand_infos:
                    hand_info = hand_infos[side]
                    hand_verts = hand_info["verts"]
                    corresp = self.mano_corresp[f"{side}_hand"]
                    human_verts[corresp] = hand_verts
                    verts_confs[corresp] = 1
            sample_masks.append(fit_info["mask"])
            sample_verts.append(human_verts)
            sample_confs.append(verts_confs)
            sample_imgs.append(img)
        fit_data = {
            "masks": torch.stack(sample_masks),
            "verts": torch.Tensor(np.stack(sample_verts)),
            "verts_confs": torch.Tensor(np.stack(sample_confs)),
            "imgs": sample_imgs,
        }
        return fit_data

    def get_optim_params(self, optim_shape=False, batch_size=1):
        smpl_model = self.smpl_model
        smpl_model.global_orient.requires_grad = False
        smpl_model.transl.requires_grad = False
        smpl_model.leye_pose.requires_grad = False
        smpl_model.reye_pose.requires_grad = False
        smpl_model.jaw_pose.requires_grad = False
        smpl_model.expression.requires_grad = False
        smpl_model.body_pose.requires_grad = False
        if not optim_shape:
            smpl_model.betas.requires_grad = False

        # Check that all gradients are set correctly
        params_to_optim = {
            name: param
            for name, param in smpl_model.named_parameters()
            if param.requires_grad
        }
        optim_params = ["left_hand_pose", "right_hand_pose"]
        if optim_shape:
            optim_params.append("betas")
        assert sorted(list(params_to_optim.keys())) == sorted(optim_params)
        pose_embedding = (
            self.get_neutral_pose_embedding()
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        left_hand_pose = smpl_model.left_hand_pose.repeat(
            batch_size, 1
        ).detach()
        right_hand_pose = smpl_model.right_hand_pose.repeat(
            batch_size, 1
        ).detach()
        left_hand_pose.requires_grad = True
        left_hand_pose.requires_grad = True
        pose_embedding.requires_grad = True

        optim_params = {}
        optim_params["pose_embedding"] = pose_embedding
        optim_params["left_hand_pose"] = left_hand_pose
        optim_params["right_hand_pose"] = right_hand_pose
        return optim_params

    def get_neutral_pose_embedding(self):
        pose_embedding = torch.Tensor(
            [
                4.4791e-02,
                -2.4548e-01,
                5.0720e-02,
                2.2368e-02,
                1.6534e-02,
                1.0907e-02,
                -2.4805e-01,
                -8.7074e-02,
                2.0321e-01,
                3.6301e-01,
                -2.6616e-01,
                9.5585e-02,
                1.0855e00,
                -9.1609e-03,
                -4.0064e-01,
                8.0400e-02,
                -7.0391e-02,
                -4.7131e-01,
                8.3280e-04,
                -9.0358e-02,
                7.6938e-01,
                -1.6985e-02,
                -2.2199e00,
                -8.9812e-02,
                -8.9908e-02,
                3.3213e-02,
                -4.8088e-01,
                -1.9684e-02,
                -1.7459e-01,
                1.1451e-01,
                -3.0125e-01,
                -1.2514e-01,
            ]
        )
        return pose_embedding
