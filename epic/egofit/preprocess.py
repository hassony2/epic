import pickle

import pandas as pd
import numpy as np
import torch

from epic.io.tarutils import TarReader
from kornia.geometry.camera import perspective

from epic.rendering import py3drendutils
from libyana.renderutils import catmesh


def lift_verts(verts, camintr):
    unproj3d = perspective.unproject_points(
        verts[:, :, :2], verts[:, :, 2:] / 200 + 0.5, camintr
    )
    return unproj3d


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
        ref_hand_rends = []

        # Create dummy intrinsic camera for supervision rendering
        focal = 200
        camintr = np.array(
            [[focal, 0, 456 // 2], [0, focal, 256 // 2], [0, 0, 1]]
        )
        camintr_th = torch.Tensor(camintr).unsqueeze(0)
        for fit_info in fit_infos:
            img = tareader.read_tar_frame(fit_info["img_path"])
            img_size = img.shape[:2]  # height, width
            # img = cv2.imread(fit_info["img_path"])
            hand_infos = fit_info["hands"]
            human_verts = np.zeros((self.smplx_vertex_nb, 3))
            verts_confs = np.zeros((self.smplx_vertex_nb,))

            # Get hand vertex refernces poses
            img_hand_verts = []
            img_hand_faces = []
            for side in hand_infos:
                hand_info = hand_infos[side]
                hand_verts = hand_info["verts"]
                # Aggregate hand vertices and faces for rendering
                img_hand_verts.append(
                    lift_verts(
                        torch.Tensor(hand_verts).unsqueeze(0), camintr_th
                    )
                )
                img_hand_faces.append(
                    torch.Tensor(hand_info["faces"]).unsqueeze(0)
                )
                corresp = self.mano_corresp[f"{side}_hand"]
                human_verts[corresp] = hand_verts
                verts_confs[corresp] = 1

            # render reference hands
            img_hand_verts, img_hand_faces, _ = catmesh.batch_cat_meshes(
                img_hand_verts, img_hand_faces
            )
            if len(img_hand_verts):
                with torch.no_grad():
                    res = py3drendutils.batch_render(
                        img_hand_verts.cuda(),
                        img_hand_faces.cuda(),
                        faces_per_pixel=2,
                        color=(1, 0.4, 0.6),
                        K=camintr_th,
                        image_sizes=[(img_size[1], img_size[0])],
                    )
                ref_hand_rends.append(res[0, :, :, :3].cpu().numpy())
            else:
                ref_hand_rends.append(np.zeros(img.shape) + 1)

            sample_masks.append(fit_info["mask"])
            sample_verts.append(human_verts)
            sample_confs.append(verts_confs)
            sample_imgs.append(img)
            verts = torch.Tensor(np.stack(sample_verts))
        fit_data = {
            "masks": torch.stack(sample_masks),
            "verts": verts,
            "verts_confs": torch.Tensor(np.stack(sample_confs)),
            "imgs": sample_imgs,
            "ref_hand_rends": ref_hand_rends,
        }
        return fit_data
