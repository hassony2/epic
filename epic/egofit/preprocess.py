import pickle

import pandas as pd
import numpy as np
import torch

from epic.io.tarutils import TarReader


class Preprocessor:
    def __init__(
        self, mano_corresp_path="assets/models/MANO_SMPLX_vertex_ids.pkl"
    ):
        self.smplx_vertex_nb = 10475
        with open(mano_corresp_path, "rb") as p_f:
            self.mano_corresp = pickle.load(p_f)

    def preprocess_df(self, fit_infos):
        fit_infos_df_keys = ["frame_idx", "video_id", "img_path"]
        df_dicts = []
        for fit_info in fit_infos:
            df_dict = {key: fit_info[key] for key in fit_infos_df_keys}
            df_dict["obj_paths"] = [
                fit_info["obj_path"],
            ]
            df_dict["boxes"] = fit_info["boxes"].cpu().numpy()
            df_dicts.append(df_dict)
        return pd.DataFrame(df_dicts)

    def preprocess_supervision(self, fit_infos):
        # Initialize tar reader
        tareader = TarReader()
        sample_masks = []
        sample_verts = []
        sample_confs = []
        sample_imgs = []
        for fit_info in fit_infos:
            img = tareader.read_tar_frame(fit_info["img_path"])
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
