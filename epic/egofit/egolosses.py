import numpy as np
import torch
from torch.nn import functional as torch_f
from pytorch3d.ops.knn import knn_points

from robust_loss_pytorch.adaptive import AdaptiveLossFunction

from libyana.metrics import iou as lyiou
from libyana.conversions import npt


class EgoLosses:
    def __init__(
        self,
        lambda_hand_v=1,
        loss_hand_v="l1",
        lambda_link=1,
        loss_link="l1",
        lambda_obj_mask=1,
        loss_obj_mask="l1",
        lambda_obj_smooth=1,
        loss_obj_smooth="l1",
        lambda_body_smooth=1,
        loss_body_smooth="l1",
        norm_hand_v=100,
        render_size=(256, 256),
        obj_nb=2,
        mask_mode="mask",
    ):
        # Hand supervision parameters
        self.lambda_hand_v = lambda_hand_v
        self.loss_hand_v = loss_hand_v
        self.norm_hand_v = norm_hand_v
        self.lambda_link = lambda_link
        # Rendering supervision parameters
        self.lambda_obj_mask = lambda_obj_mask
        self.loss_obj_mask = loss_obj_mask
        self.mask_mode = mask_mode

        # Smoothness supervision parameters
        self.lambda_obj_smooth = lambda_obj_smooth
        self.loss_obj_smooth = loss_obj_smooth
        self.lambda_body_smooth = lambda_body_smooth
        self.loss_body_smooth = loss_body_smooth

        self.obj_nb = obj_nb

        if self.mask_mode == "mask":
            self.mask_adaptive_loss = AdaptiveLossFunction(
                num_dims=(1 * render_size[0] * render_size[1]),
                float_dtype=np.float32,
                device="cuda:0",
            )
        elif self.mask_mode in ["segm", "segmask"]:
            self.mask_adaptive_loss = AdaptiveLossFunction(
                num_dims=((obj_nb - 1) * render_size[0] * render_size[1]),
                float_dtype=np.float32,
                device="cuda:0",
            )
        else:
            raise ValueError(f"{self.mask_mode} not in [mask|segm]")

    def get_optim_params(self):
        if self.loss_obj_mask == "adapt":
            return list(self.mask_adaptive_loss.parameters())
        else:
            return []

    def compute_losses(self, scene_outputs, supervision):
        loss_meta = {}
        hand_v_loss = self.compute_hand_v_loss(scene_outputs, supervision)
        obj_mask_loss, obj_mask_meta = self.compute_obj_mask_loss(
            scene_outputs, supervision
        )
        loss_meta["mask_diffs"] = obj_mask_meta["mask_diffs"]
        link_loss, _ = self.compute_link_loss(scene_outputs, supervision)
        obj_smooth_loss = self.compute_obj_smooth_loss(scene_outputs)
        body_smooth_loss = self.compute_body_smooth_loss(scene_outputs)
        loss = (
            hand_v_loss * self.lambda_hand_v
            + obj_mask_loss * self.lambda_obj_mask
            + link_loss * self.lambda_link
            + obj_smooth_loss * self.lambda_obj_smooth
            + body_smooth_loss * self.lambda_body_smooth
        )
        losses = {
            "hand_v": hand_v_loss.item(),
            "obj_mask": obj_mask_loss.item(),
            "obj_smooth": obj_smooth_loss.item(),
            "body_smooth": body_smooth_loss.item(),
            "link": link_loss.item(),
            "loss": loss.item(),
        }
        return loss, losses, loss_meta

    def compute_body_smooth_loss(self, scene_output):
        human_verts = scene_output["body_info"]["verts"]
        vert_time_offsets = human_verts[1:] - human_verts[:-1]
        if self.loss_obj_smooth == "l1":
            loss = vert_time_offsets.abs().mean()
        elif self.loss_obj_smooth == "l2":
            loss = (vert_time_offsets ** 2).sum(-1).mean()
        return loss

    def compute_obj_smooth_loss(self, scene_output):
        loss = 0
        for obj in scene_output["obj_infos"]:
            obj_verts = obj["verts"]  # (time, vert_nb, 3)
            # Differences between consecutive time steps
            vert_time_offsets = obj_verts[1:] - obj_verts[:-1]
            if self.loss_obj_smooth == "l1":
                loss += vert_time_offsets.abs().mean()
            elif self.loss_obj_smooth == "l2":
                loss += (vert_time_offsets ** 2).sum(-1).mean()
        return loss

    def compute_link_loss(self, scene_output, supervision):
        corresp = supervision["mano_corresp"]
        links = supervision["links"]
        body_verts = scene_output["body_info"]["verts"]
        left_hand_verts = body_verts[:, corresp["left_hand"]]
        right_hand_verts = body_verts[:, corresp["right_hand"]]
        obj_verts = torch.cat(
            [info["verts"] for info in scene_output["obj_infos"]], 2
        )

        # Compute min obj2 hand distance
        left2obj_mins = knn_points(obj_verts, left_hand_verts, K=1)[0].min(1)[
            0
        ][:, 0]
        right2obj_mins = knn_points(obj_verts, right_hand_verts, K=1)[0].min(
            1
        )[0][:, 0]
        right_flags = obj_verts.new(links)[:, 1]
        left_flags = obj_verts.new(links)[:, 0]

        batch_min_dists = (
            left_flags * left2obj_mins + right_flags * right2obj_mins
        )
        loss = batch_min_dists.mean()
        min_dists = {
            "left": left2obj_mins.detach().cpu(),
            "right": right2obj_mins.detach().cpu(),
            "left_flags": left_flags.detach().cpu(),
            "right_flags": right_flags.detach(),
        }
        loss_info = {"link_min_dists": min_dists}
        return loss, loss_info

    def compute_metrics(self, scene_outputs, supervision):
        # Get hand vertex information
        pred_verts = scene_outputs["body_info"]["verts2d"]
        gt_verts = supervision["verts"].to(pred_verts.device)[:, :, :2]
        weights_verts = supervision["verts_confs"].to(pred_verts.device)

        # Compute per-vertex pixel distances
        diffs = (pred_verts - gt_verts).norm(2, -1)
        dists = (diffs * weights_verts).sum() / weights_verts.sum()

        # Compute mask IoU
        pred_masks = scene_outputs["segm_rend"]
        pseudo_obj_masks = (
            supervision["objs_masks_crops"]
            .permute(0, 2, 3, 1)
            .to(pred_masks.device)
        )
        mask_iou = lyiou.batch_mask_iou(
            (pred_masks[:, :, :, 1:] > 0), (pseudo_obj_masks > 0)
        ).mean()
        return {"hand_v_dists": dists.item(), "obj_mask_iou": mask_iou.item()}

    def compute_hand_v_loss(self, scene_outputs, supervision):
        pred_verts = scene_outputs["body_info"]["verts2d"] / self.norm_hand_v
        gt_verts = (
            supervision["verts"].to(pred_verts.device)[:, :, :2]
            / self.norm_hand_v
        )
        if self.loss_hand_v == "l1":
            errs = torch_f.l1_loss(
                gt_verts, pred_verts, reduction="none"
            ).mean(-1)
        if self.loss_hand_v == "l2":
            errs = torch_f.mse_loss(
                gt_verts, pred_verts, reduction="none"
            ).sum(-1)
        weights_verts = supervision["verts_confs"].to(pred_verts.device)
        loss = (errs * weights_verts).sum() / (weights_verts).sum()
        return loss

    def compute_obj_mask_loss(self, scene_outputs, supervision):
        rend = scene_outputs["segm_rend"]
        device = rend.device
        gt_obj_masks = (
            supervision["objs_masks_crops"].permute(0, 2, 3, 1).to(device)
        )
        gt_hand_masks = supervision["hand_masks_crops"].to(device)
        if self.mask_mode == "segm":
            pred_masks = rend[:, :, :, :-1]  # Remove alpha channel
            gt_masks = torch.cat(
                [gt_hand_masks.unsqueeze(-1), gt_obj_masks], -1
            )
            sup_masks = (
                (
                    (gt_hand_masks.unsqueeze(-1).sum([1, 2, 3]) > 0)
                    & (gt_obj_masks.sum([1, 2, 3]) > 0)
                )
                .float()
                .view(-1, 1, 1, 1)
            )
            optim_mask_diff = gt_masks - pred_masks
            optim_mask_diff = optim_mask_diff * sup_masks
            if self.loss_obj_mask == "l1":
                loss = optim_mask_diff.abs().mean()
            elif self.loss_obj_mask == "l2":
                loss = (optim_mask_diff ** 2).mean()
            elif self.loss_obj_mask == "adapt":
                loss = self.mask_adaptive_loss.lossfun(
                    optim_mask_diff.view(gt_masks.shape[0], -1)
                ).mean()
            masked_diffs = npt.numpify(gt_masks) - npt.numpify(pred_masks)
        elif self.mask_mode == "segmask":
            pred_masks = rend[:, :, :, :-1]  # Remove alpha channel
            gt_masks = torch.cat(
                [gt_hand_masks.unsqueeze(-1), gt_obj_masks], -1
            )
            # Get region to penalize by computing complementary from gt masks
            comp_obj_idxs = [
                [idx for idx in range(self.obj_nb) if idx != obj_idx]
                for obj_idx in range(self.obj_nb)
            ]
            sup_mask = torch.cat(
                [
                    1
                    - gt_masks[:, :, :, comp_idxs]
                    .sum(-1, keepdim=True)
                    .clamp(0, 1)
                    for comp_idxs in comp_obj_idxs
                ],
                -1,
            )
            sup_mask = (
                (gt_hand_masks.unsqueeze(-1).sum([1, 2, 3]) > 0)
                & (gt_obj_masks.sum([1, 2, 3]) > 0)
            ).float().view(-1, 1, 1, 1) * sup_mask
            masked_diffs = sup_mask * (gt_masks - pred_masks)
            if self.loss_obj_mask == "l1":
                loss = masked_diffs.abs().sum() / sup_mask.sum()
            elif self.loss_obj_mask == "l2":
                loss = (masked_diffs ** 2).sum() / sup_mask.sum()
            elif self.loss_obj_mask == "adapt":
                loss = self.mask_adaptive_loss.lossfun(
                    masked_diffs.view(sup_mask.shape[0], -1)
                ).mean()
        elif self.mask_mode == "mask":
            pred_obj_masks = rend[:, :, :, 1:-1]
            obj_mask_diffs = gt_obj_masks[:, :, :, :] - pred_obj_masks
            if obj_mask_diffs.shape[-1] != 1:
                raise NotImplementedError("No handling of multiple objects")
            sup_mask = (1 - gt_hand_masks).unsqueeze(-1)
            # Zero supervision on frames which do not have both hand and masks
            sup_mask = (
                (gt_hand_masks.unsqueeze(-1).sum([1, 2, 3]) > 0)
                & (gt_obj_masks.sum([1, 2, 3]) > 0)
            ).float().view(-1, 1, 1, 1) * sup_mask

            masked_diffs = sup_mask * obj_mask_diffs
            if self.loss_obj_mask == "l2":
                loss = (masked_diffs ** 2).sum() / sup_mask.sum()
            elif self.loss_obj_mask == "l1":
                loss = (masked_diffs.abs()).sum() / sup_mask.sum()
            elif self.loss_obj_mask == "adapt":
                loss = self.mask_adaptive_loss.lossfun(
                    masked_diffs.view(sup_mask.shape[0], -1)
                ).mean()
        return (loss, {"mask_diffs": masked_diffs})
