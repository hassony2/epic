import numpy as np
import torch

from epic.egofit.egohuman import EgoHuman
from epic.egofit.manobj import ManipulatedObject
from epic.rendering.py3drendutils import batch_render, get_colors

from libyana.renderutils import catmesh


def get_segm_colors(vert_list, faces_list):
    obj_nb = len(vert_list) - 1
    batch_size = vert_list[0].shape[0]
    if len(vert_list) != len(faces_list):
        raise ValueError(
            f"Expected same number of verts {obj_nb} and faces {len(faces_list)}"
        )
    faces_off_idxs = np.cumsum([face.shape[1] for face in faces_list]).tolist()
    segm_onehots = vert_list[0].new_zeros(
        batch_size, faces_off_idxs[-1], obj_nb
    )
    faces_off_idxs.append(-1)
    for obj_idx in range(obj_nb):
        # Skipe body vertices
        obj_start_idx = faces_off_idxs[obj_idx]
        obj_end_idx = faces_off_idxs[obj_idx + 1]
        segm_onehots[:, obj_start_idx:obj_end_idx, obj_idx] = 1
    return segm_onehots


class Scene:
    def __init__(
        self,
        data_df,
        camera,
        hand_pca_nb=6,
        vposer_dim=32,
        debug=True,
    ):
        batch_size = len(data_df)
        self.batch_size = batch_size
        self.data_df = data_df
        all_obj_paths = np.array(data_df.obj_paths.tolist()).transpose()
        for obj_paths in all_obj_paths:
            matches_first_obj_path = [
                obj_path == obj_paths[0] for obj_path in obj_paths
            ]
            if not np.all(matches_first_obj_path):
                raise ValueError(
                    f"Expected all object paths to be equal, but got {all_obj_paths}"
                )
        obj_bboxes = np.stack(data_df.boxes.to_list()).transpose(
            1, 0, 2
        )  # (objects, time, 4)
        self.egohuman = EgoHuman(
            batch_size=batch_size,
            hand_pca_nb=hand_pca_nb,
            vposer_dim=vposer_dim,
        )
        self.objects = [
            ManipulatedObject(
                obj_paths[0],
                bboxes=bboxes,
                camintr=camera.camintr,
                camextr=camera.camextr,
            )
            for obj_paths, bboxes in zip(all_obj_paths, obj_bboxes)
        ]
        self.camera = camera
        self.debug = debug

    def cuda(self):
        self.egohuman.cuda()
        for obj in self.objects:
            obj.cuda()
        self.camera.cuda()

    def cpu(self):
        self.egohuman.cpu()
        for obj in self.objects:
            obj.cpu()
        self.camera.cpu()

    def get_optim_params(self):
        # Collect EgoHuman parameters
        params = self.egohuman.get_params()

        # Aggregate object parameters
        for obj in self.objects:
            obj_params = obj.get_params()
            for obj_param in obj_params:
                params.append(obj_param)
        return params

    def reset_obj2hand(self):
        handverts = self.egohuman.get_handverts()
        camintr = self.camera.camintr.unsqueeze(0).repeat(
            self.batch_size, 1, 1
        )
        camextr = self.camera.camextr.unsqueeze(0).repeat(
            self.batch_size, 1, 1
        )
        for obj in self.objects:
            hand_links = obj.hand_links
            z_offs = []
            for sample_idx, sides in enumerate(hand_links):
                verts = [handverts[side][sample_idx] for side in sides]
                z_off = torch.cat(verts)[:, 2].median()
                z_offs.append(z_off)
            z_offs = torch.stack(z_offs)
            obj.init_scale_trans_from_depth(z_offs, camintr, camextr)

    def forward(self, faces_per_pixel=10, viz_views=True):
        body_info = self.egohuman.forward()
        obj_infos = [obj.forward() for obj in self.objects]
        body_verts = body_info["verts"]
        obj_verts = [obj_info["verts"] for obj_info in obj_infos]
        verts = [body_verts] + obj_verts
        faces = [body_info["faces"]] + [
            obj_info["faces"] for obj_info in obj_infos
        ]
        verts2d = self.camera.project(body_info["verts"])
        body_info["verts2d"] = verts2d

        camintr = self.camera.camintr.to(verts2d.device).unsqueeze(0)
        rot = self.camera.rot.to(verts2d.device).unsqueeze(0)
        trans = self.camera.trans.to(verts2d.device).unsqueeze(0)
        height, width = self.camera.image_size

        body_color = ((0.25, 0.73, 1),)  # light_blue
        # obj_color = (0.74117647, 0.85882353, 0.65098039),  # light_green
        obj_color = ((0.25, 0.85, 0.85),)  # light_blue_green
        viz_colors = [get_colors(body_verts, body_color)] + [
            get_colors(obj_vert, obj_color) for obj_vert in obj_verts
        ]
        all_verts, all_faces, all_viz_colors = catmesh.batch_cat_meshes(
            verts, faces, viz_colors
        )
        face_colors = get_segm_colors(verts, faces)
        rendres = batch_render(
            all_verts,
            all_faces,
            K=camintr,
            rot=rot,
            trans=trans,
            image_sizes=[(width, height)],
            shading="facecolor",
            face_colors=face_colors,
            faces_per_pixel=faces_per_pixel,
        )
        scene_res = {
            "body_info": body_info,
            "obj_infos": obj_infos,
            "segm_rend": rendres,
        }
        if viz_views:
            with torch.no_grad():
                # Render scene in camera view
                cam_rendres = batch_render(
                    all_verts,
                    all_faces,
                    colors=all_viz_colors,
                    K=camintr,
                    rot=rot,
                    trans=trans,
                    image_sizes=[(width, height)],
                    mode="rgb",
                    faces_per_pixel=faces_per_pixel,
                )
                viz_verts = all_verts.clone()
                viz_verts[:, :, 2] = -viz_verts[:, :, 2]
                front_rendres = batch_render(
                    all_verts.new([0, 0, 0.7]) + viz_verts,
                    torch.flip(all_faces, (2,)),  # Compensate for - in verts
                    colors=all_viz_colors,
                    K=camintr,
                    rot=rot,
                    trans=trans,
                    image_sizes=[(width, height)],
                    mode="rgb",
                    faces_per_pixel=faces_per_pixel,
                )
            scene_res["scene_viz_rend"] = [cam_rendres, front_rendres]
        return scene_res

    def __len__(self):
        return len(self.data_df)

    def __repr__(self):
        return f"egoscene: {len(self.objects)} objects, {len(self)} scenes"
