import torch
from torch.nn import functional as torch_f
from pytorch3d.ops.knn import knn_points

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
        norm_hand_v=100,
    ):
        self.lambda_hand_v = lambda_hand_v
        self.loss_hand_v = loss_hand_v
        self.norm_hand_v = norm_hand_v
        self.lambda_link = lambda_link
        self.lambda_obj_mask = lambda_obj_mask
        self.loss_obj_mask = loss_obj_mask

    def compute_losses(self, scene_outputs, supervision):
        loss_meta = {}
        hand_v_loss = self.compute_hand_v_loss(scene_outputs, supervision)
        obj_mask_loss, obj_mask_meta = self.compute_obj_mask_loss(
            scene_outputs, supervision
        )
        loss_meta["mask_diffs"] = obj_mask_meta["mask_diffs"]
        link_loss, _ = self.compute_link_loss(scene_outputs, supervision)
        loss = (
            hand_v_loss * self.lambda_hand_v
            + obj_mask_loss * self.lambda_obj_mask
            + link_loss * self.lambda_link
        )
        losses = {
            "hand_v": hand_v_loss.item(),
            "obj_mask": obj_mask_loss.item(),
            "link": link_loss.item(),
            "loss": loss.item(),
        }
        return loss, losses, loss_meta

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
        pred_masks = scene_outputs["segm_rend"]
        gt_obj_masks = (
            supervision["objs_masks_crops"]
            .permute(0, 2, 3, 1)
            .to(pred_masks.device)
        )
        gt_hand_masks = supervision["hand_masks_crops"].to(pred_masks.device)
        gt_masks = torch.cat([gt_hand_masks.unsqueeze(-1), gt_obj_masks], -1)
        if self.loss_obj_mask == "l1":
            loss = torch_f.l1_loss(
                gt_masks, pred_masks, reduction="none"
            ).mean()
        else:
            loss = torch_f.mse_loss(
                gt_masks, pred_masks, reduction="none"
            ).mean()
        return loss, {
            "mask_diffs": npt.numpify(gt_masks) - npt.numpify(pred_masks)
        }
