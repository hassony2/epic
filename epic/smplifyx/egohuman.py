from copy import deepcopy
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from human_body_prior.tools.model_loader import load_vposer
from epic.smplifyx import (
    camera,
    prior,
    initialize,
    vposeutils,
    fitting,
    optim_factory,
    smplvis,
)
from epic.io.tarutils import TarReader


class EgoHuman:
    def __init__(
        self,
        debug=True,
        camintr=None,
        hand_pca_nb=6,
        head_center_idx=8949,
        opt_weights={
            # "hand_prior_weight":[1e2, 5 * 1e1, 1e1, 0.5 * 1e1],
            "hand_prior_weight": [0, 0, 0, 0],
            "hand_weight": [0.0, 0.0, 0.0, 1.0],
            # "body_pose_weight": [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78],
            "body_pose_weight": [0, 0, 0, 0],
            # "shape_weight": [1e2, 5 * 1e1, 1e1, 0.5 * 1e1]},
            "shape_weight": [0, 0, 0, 0],
        },
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

        # Optim weights
        self.opt_weights = opt_weights
        self.data_weight = data_weight

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
        self.joint_weights = initialize.get_joint_weights()
        fx, fy = camintr[0, 0], camintr[1, 1]
        center = camintr[:2, 2]
        rot = torch.eye(3)
        rot[0, 0] = -1
        rot[1, 1] = -1
        self.camera = camera.create_camera(
            focal_length_x=fx,
            focal_length_y=fy,
            center=torch.Tensor(center).unsqueeze(0),
            rotation=rot.unsqueeze(0),
        )

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

    def preprocess(self, fit_infos):
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

    def prepare_weights(self):
        weight_len = len(list(self.opt_weights.values())[0])
        weights = []
        for weight_idx in range(weight_len):
            opt_weights = {
                key: val[weight_idx] for key, val in self.opt_weights.items()
            }
            opt_weights["data_weight"] = self.data_weight
            opt_weights["bending_prior_weight"] = (
                3.17 * opt_weights["body_pose_weight"]
            )
            weights.append(opt_weights)
        if self.debug:
            print(weights)
        return weights

    def set_head2cam_trans(self):
        pose_embedding = torch.zeros([1, self.vposer_dim])
        vpose = vposeutils.get_vposer_pose(
            self.smpl_model, self.vposer, pose_embedding
        )
        smplx_verts = self.smpl_model(
            betas=self.smpl_model.betas, body_pose=vpose, return_verts=True
        ).vertices
        head_loc = smplx_verts[:, self.head_center_idx]
        self.smpl_model.transl.data[0] = -head_loc

        vpose = vposeutils.get_vposer_pose(
            self.smpl_model, self.vposer, pose_embedding
        )
        smplx_verts = self.smpl_model(
            betas=self.smpl_model.betas, body_pose=vpose, return_verts=True
        ).vertices

    def fit(
        self,
        fit_infos,
        iters=200,
        lr=0.01,
        save_root="tmp/optim/",
        optimizer="adam",
    ):
        save_folder = (
            Path(save_root) / f"opt{optimizer}_lr{lr:.4f}_it{iters:04d}"
        )
        save_folder.mkdir(exist_ok=True, parents=True)
        batch_size = len(fit_infos)
        fit_data = self.preprocess(fit_infos)
        pose_embedding = torch.Tensor(
            [
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
            ]
        ).repeat(batch_size, 1)
        pose_embedding.requires_grad = True

        loss = fitting.create_loss(
            loss_type="smplify",
            angle_prior=self.angle_prior,
            body_pose_prior=self.body_pose_prior,
            left_hand_prior=self.left_hand_prior,
            pose_embedding=pose_embedding,
            rho=100,
            right_hand_prior=self.right_hand_prior,
            shape_prior=self.shape_prior,
            use_face=False,
            use_hands=True,
            use_vposer=True,
            vert_weights=self.opt_weights["hand_weight"],
            vposer=self.vposer,
        )
        opt_weights = self.prepare_weights()
        with fitting.FittingMonitor(
            batch_size=batch_size,
            visualize=False,
            model_type="smplx",
            maxiters=iters,
        ) as monitor:
            smpl_model = deepcopy(self.smpl_model)
            smpl_model.global_orient.requires_grad = False
            smpl_model.transl.requires_grad = False
            smpl_model.leye_pose.requires_grad = False
            smpl_model.reye_pose.requires_grad = False
            smpl_model.jaw_pose.requires_grad = False
            smpl_model.expression.requires_grad = False
            smpl_model.body_pose.requires_grad = False
            # TODO remove ?
            smpl_model.betas.requires_grad = False
            for opt_idx, curr_weights in enumerate(
                tqdm(opt_weights, desc="opt stage")
            ):
                params_to_optim = sorted(
                    [
                        name
                        for name, param in smpl_model.named_parameters()
                        if param.requires_grad
                    ]
                )
                assert params_to_optim == ["left_hand_pose", "right_hand_pose"]
                smpl_params = [
                    param
                    for param in smpl_model.parameters()
                    if param.requires_grad
                ]
                smpl_params.append(pose_embedding.to(smpl_params[0].device))

                # Initialize optimizer
                (
                    body_optimizer,
                    body_create_graph,
                ) = optim_factory.create_optimizer(
                    smpl_params, optim_type=optimizer, lr=lr
                )
                body_optimizer.zero_grad()
                loss.reset_loss_weights(curr_weights)
                closure = monitor.create_fitting_closure(
                    body_optimizer,
                    smpl_model,
                    body_create_graph=body_create_graph,
                    camera=self.camera,
                    gt_verts=fit_data["verts"],
                    verts_conf=fit_data["verts_confs"],
                    vert_weights=fit_data["verts_confs"],
                    imgs=fit_data["imgs"],
                    loss=loss,
                    pose_embedding=pose_embedding,
                    return_verts=True,
                    return_full_pose=True,
                    show_faces=self.armfaces,
                    model_faces=self.smpl_f,
                    use_vposer=True,
                    vposer=self.vposer,
                )
                fit_res = monitor.run_fitting(
                    body_optimizer,
                    closure,
                    smpl_params,
                    smpl_model,
                    pose_embedding=pose_embedding,
                    vposer=self.vposer,
                    use_vposer=True,
                )
                left_hand_pose = smpl_model.left_hand_pose.detach().cpu()
                right_hand_pose = smpl_model.right_hand_pose.detach().cpu()
                body_pose = smpl_model.body_pose.detach().cpu()
                res = {
                    "left_pose": left_hand_pose,
                    "right_pose": right_hand_pose,
                    "body_pose": body_pose,
                    "fit_res": fit_res,
                    "fit_data": fit_data,
                }
                with (save_folder / "res.pkl").open("wb") as p_f:
                    pickle.dump(res, p_f)

                import pdb

                pdb.set_trace()
