import pickle

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from epic.io.tarutils import TarReader
from kornia.geometry.camera import perspective
from skimage.morphology import skeletonize

from epic.lib2d import grabcut, boxutils, cropping
from epic.rendering import py3drendutils
from libyana.renderutils import catmesh
from libyana.conversions import npt


def lift_verts(verts, camintr):
    unproj3d = perspective.unproject_points(
        verts[:, :, :2], verts[:, :, 2:] / 200 + 0.5, camintr
    )
    return unproj3d


def preprocess_links(links):
    """
    returns [0 1] which encodes for each hand if a link to objects exists
    """
    link_rights = [link for link in links if link["side"] == "right"]
    link_right = 1 if len(link_rights) else 0
    link_lefts = [link for link in links if link["side"] == "right"]
    link_left = 1 if len(link_lefts) else 0
    links = [link_left, link_right]
    return links


class Preprocessor:
    def __init__(
        self,
        mano_corresp_path="assets/models/MANO_SMPLX_vertex_ids.pkl",
        crop_size=(256, 256),
        debug=False,
    ):
        self.debug = debug
        self.smplx_vertex_nb = 10475
        self.crop_size = crop_size
        with open(mano_corresp_path, "rb") as p_f:
            self.mano_corresp = pickle.load(p_f)

    def preprocess_df(self, fit_infos):
        fit_infos_df_keys = ["frame_idx", "video_id", "img_path"]
        df_dicts = []
        for fit_info in fit_infos:
            df_dict = {key: fit_info[key] for key in fit_infos_df_keys}
            df_dict["obj_paths"] = [fit_info["obj_path"]]
            df_dict["boxes"] = npt.numpify(fit_info["boxes"])
            df_dicts.append(df_dict)
        return pd.DataFrame(df_dicts)

    def preprocess_supervision(self, fit_infos, grab_objects=False):
        # Initialize tar reader
        tareader = TarReader()
        # sample_masks = []
        sample_verts = []
        sample_confs = []
        sample_imgs = []
        ref_hand_rends = []
        # Regions of interest containing hands and objects
        roi_bboxes = []
        roi_valid_masks = []
        # Crops of hand and object masks
        sample_hand_masks = []
        sample_objs_masks = []

        # Create dummy intrinsic camera for supervision rendering
        focal = 200
        camintr = np.array(
            [[focal, 0, 456 // 2], [0, focal, 256 // 2], [0, 0, 1]]
        )
        camintr_th = torch.Tensor(camintr).unsqueeze(0)
        # Modelling hand color
        print("Preprocessing sequence")
        for fit_info in tqdm(fit_infos):
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

            has_hands = len(img_hand_verts) > 0
            # render reference hands
            if has_hands:
                img_hand_verts, img_hand_faces, _ = catmesh.batch_cat_meshes(
                    img_hand_verts, img_hand_faces
                )
                with torch.no_grad():
                    res = py3drendutils.batch_render(
                        img_hand_verts.cuda(),
                        img_hand_faces.cuda(),
                        faces_per_pixel=2,
                        color=(1, 0.4, 0.6),
                        K=camintr_th,
                        image_sizes=[(img_size[1], img_size[0])],
                        mode="rgb",
                        shading="soft",
                    )
                ref_hand_rends.append(npt.numpify(res[0, :, :, :3]))
                hand_mask = npt.numpify(res[0, :, :, 3])
            else:
                ref_hand_rends.append(np.zeros(img.shape) + 1)
                hand_mask = np.zeros((img.shape[:2]))
            obj_masks = fit_info["masks"]
            # GrabCut objects
            has_objs = len(obj_masks) > 0
            if has_objs:
                obj_masks_aggreg = (
                    npt.numpify(torch.stack(obj_masks)).sum(0) > 0
                )
            else:
                obj_masks_aggreg = np.zeros_like(hand_mask)
            # Detect if some pseudo ground truth masks exist
            has_both_masks = (hand_mask.max() > 0) and (
                obj_masks_aggreg.max() > 0
            )
            if has_both_masks:
                xs, ys = np.where((hand_mask + obj_masks_aggreg) > 0)
                # Compute region of interest which contains hands and objects
                roi_bbox = boxutils.squarify_box(
                    [xs.min(), ys.min(), xs.max(), ys.max()], scale_factor=1.5
                )
            else:
                rad = min(img.shape[:2])
                roi_bbox = [0, 0, rad, rad]

            roi_bbox = [int(val) for val in roi_bbox]
            roi_bboxes.append(roi_bbox)
            img_crop = cropping.crop_cv2(img, roi_bbox, resize=self.crop_size)

            # Compute region of crop which belongs to original image (vs paddding)
            roi_valid_mask = cropping.crop_cv2(
                np.ones(img.shape[:2]), roi_bbox, resize=self.crop_size
            )
            roi_valid_masks.append(roi_valid_mask)

            # Crop hand and object image
            hand_mask_crop = (
                cropping.crop_cv2(hand_mask, roi_bbox, resize=self.crop_size)
                > 0
            ).astype(np.int)
            objs_masks_crop = cropping.crop_cv2(
                obj_masks_aggreg.astype(np.int),
                roi_bbox,
                resize=self.crop_size,
            ).astype(np.int)

            # Remove object region from hand mask
            hand_mask_crop[objs_masks_crop > 0] = 0
            # Extract skeletons
            skel_objs_masks_crop = skeletonize(
                objs_masks_crop.astype(np.uint8)
            )
            skel_hand_mask_crop = skeletonize(hand_mask_crop.astype(np.uint8))

            # Removing object region from hand can cancel out whole hand !
            if has_both_masks and hand_mask_crop.max():
                grabinfo = grabcut.grab_cut(
                    img_crop.astype(np.uint8),
                    mask=hand_mask_crop,
                    bbox=roi_bbox,
                    bgd_mask=skel_objs_masks_crop,
                    fgd_mask=skel_hand_mask_crop,
                    debug=self.debug,
                )
                hand_mask = grabinfo["grab_mask"]
                hand_mask[objs_masks_crop > 0] = 0
            else:
                hand_mask = hand_mask_crop
            sample_hand_masks.append(hand_mask)

            # Get crops of object masks
            obj_mask_crops = []
            for obj_mask in obj_masks:
                obj_mask_crop = cropping.crop_cv2(
                    npt.numpify(obj_mask).astype(np.int),
                    roi_bbox,
                    resize=self.crop_size,
                )
                skel_obj_mask_crop = skeletonize(
                    obj_mask_crop.astype(np.uint8)
                )
                if grab_objects:
                    raise NotImplementedError(
                        "Maybe needs also the skeleton of other objects"
                        "to be labelled as background ?"
                    )
                    grabinfo = grabcut.grab_cut(
                        img_crop,
                        mask=obj_mask_crop,
                        bbox=roi_bbox,
                        bgd_mask=skel_hand_mask_crop,
                        fgd_mask=skel_obj_mask_crop,
                        debug=self.debug,
                    )
                    obj_mask_crop = grabinfo["grab_mask"]
                obj_mask_crops.append(obj_mask_crop)
            if len(obj_mask_crops):
                sample_objs_masks.append(np.stack(obj_mask_crops))
            else:
                sample_objs_masks.append(np.zeros((1, rad, rad)))

            # Remove object region from hand mask
            # sample_masks.append(torch.stack(fit_info["masks"]))
            sample_verts.append(human_verts)
            sample_confs.append(verts_confs)
            sample_imgs.append(img)
            verts = torch.Tensor(np.stack(sample_verts))

        links = [preprocess_links(info["links"]) for info in fit_infos]
        fit_data = {
            # "masks": torch.stack(sample_masks),
            "roi_bboxes": torch.Tensor(np.stack(roi_bboxes)),
            "roi_valid_masks": torch.Tensor(np.stack(roi_valid_masks)),
            "objs_masks_crops": torch.Tensor(np.stack(sample_objs_masks)),
            "hand_masks_crops": torch.Tensor(np.stack(sample_hand_masks)),
            "verts": verts,
            "verts_confs": torch.Tensor(np.stack(sample_confs)),
            "imgs": sample_imgs,
            "ref_hand_rends": ref_hand_rends,
            "links": links,
            "mano_corresp": self.mano_corresp,
        }
        return fit_data
