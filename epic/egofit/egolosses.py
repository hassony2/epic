from torch.nn import functional as torch_f

from libyana.metrics import iou as lyiou


class EgoLosses:
    def __init__(
        self,
        lambda_hand_v=1,
        loss_hand_v="l1",
        lambda_obj_mask=1,
        loss_obj_mask="l1",
        norm_hand_v=100,
    ):
        self.lambda_hand_v = lambda_hand_v
        self.loss_hand_v = loss_hand_v
        self.norm_hand_v = norm_hand_v

        self.lambda_obj_mask = lambda_obj_mask
        self.loss_obj_mask = loss_obj_mask

    def compute_losses(self, scene_outputs, supervision):
        hand_v_loss = self.compute_hand_v_loss(scene_outputs, supervision)
        obj_mask_loss = self.compute_obj_mask_loss(scene_outputs, supervision)
        loss = (
            hand_v_loss * self.lambda_hand_v
            + obj_mask_loss * self.lambda_obj_mask
        )
        losses = {
            "hand_v": hand_v_loss.item(),
            "obj_mask": obj_mask_loss.item(),
            "loss": loss.item(),
        }
        return loss, losses

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
        pseudo_gt_masks = supervision["masks"].permute(0, 2, 3, 1)
        mask_iou = lyiou.batch_mask_iou(
            (pred_masks > 0), (pseudo_gt_masks > 0)
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
        pseudo_gt_masks = (
            supervision["masks"].permute(0, 2, 3, 1).to(pred_masks.device)
        )
        if self.loss_obj_mask == "l1":
            loss = torch_f.l1_loss(
                pseudo_gt_masks, pred_masks, reduction="none"
            ).mean()
        else:
            loss = torch_f.mse_loss(
                pseudo_gt_masks, pred_masks, reduction="none"
            ).mean()
        return loss
