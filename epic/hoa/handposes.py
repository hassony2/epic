import numpy as np
from handmocap.hand_mocap_api import HandMocap
from epic import boxutils


class HandExtractor:
    def __init__(self, hand_checkpoint, smpl_folder):
        self.hand_extractor = HandMocap(hand_checkpoint, smpl_folder)

    def hands_from_df(self, img, hoa_df, resize_factor=1):
        hand_df = hoa_df[hoa_df.det_type == "hand"]
        hand_boxes = {}

        # Keep first right and left hand found in data frame
        for hand_idx, row in hand_df.iterrows():
            box_ltrb = boxutils.dfbox_to_norm(row, resize_factor=resize_factor)
            box_ltwh = [
                box_ltrb[0],
                box_ltrb[1],
                box_ltrb[2] - box_ltrb[0],
                box_ltrb[3] - box_ltrb[1],
            ]
            box_side = row.side
            hand_key = f"{box_side}_hand"
            if hand_key not in hand_boxes:
                hand_boxes[hand_key] = np.array(box_ltwh)
        for side in ["right", "left"]:
            hand_key = f"{side}_hand"
            if hand_key not in hand_boxes:
                hand_boxes[hand_key] = None

        _, pred_hands = self.hand_extractor.regress(
            img, [hand_boxes], add_margin=True
        )
        # Retrieve first results
        ego_hands = pred_hands[0]

        pred_hands = {}
        for side in ["right", "left"]:
            hand_key = f"{side}_hand"
            if hand_key in ego_hands and (ego_hands[hand_key] is not None):
                pred_hand = ego_hands[hand_key]
                pred_hands[side] = {
                    "verts": pred_hand["pred_vertices_img"],
                    "faces": pred_hand["faces"],
                    "hand_pose": pred_hand["pred_hand_pose"],
                }
        return pred_hands
