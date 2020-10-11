import numpy as np
import torch

from epic.egofit.egohuman import EgoHuman
from epic.egofit.manobj import ManipulatedObject
from epic.rendering.py3drendutils import batch_render

from libyana.renderutils import catmesh


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
        self.data_df = data_df
        all_obj_paths = data_df.obj_paths.to_list()
        matches_first_obj_path = [
            obj_path == all_obj_paths[0] for obj_path in all_obj_paths
        ]
        if not np.all(matches_first_obj_path):
            raise ValueError(
                f"Expected all object paths to be equal, but got {all_obj_paths}"
            )
        bboxes = np.stack(data_df.boxes.to_list())
        self.egohuman = EgoHuman(
            batch_size=batch_size,
            hand_pca_nb=hand_pca_nb,
            vposer_dim=vposer_dim,
        )
        self.objects = [
            ManipulatedObject(
                obj_path,
                bboxes=bboxes,
                camintr=camera.camintr,
                camextr=camera.camextr,
            )
            for obj_path in all_obj_paths[0]
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

    def forward(self, faces_per_pixel=10, viz_views=True):
        body_info = self.egohuman.forward()
        obj_infos = [obj.forward() for obj in self.objects]
        verts = [body_info["verts"]] + [
            obj_info["verts"] for obj_info in obj_infos
        ]
        faces = [body_info["faces"]] + [
            obj_info["faces"] for obj_info in obj_infos
        ]
        verts2d = self.camera.project(body_info["verts"])
        body_info["verts2d"] = verts2d

        all_verts, all_faces, _ = catmesh.batch_cat_meshes(verts, faces)
        camintr = self.camera.camintr.to(all_verts.device).unsqueeze(0)
        rot = self.camera.rot.to(all_verts.device).unsqueeze(0)
        trans = self.camera.trans.to(all_verts.device).unsqueeze(0)
        height, width = self.camera.image_size

        # Render scene
        rendres = batch_render(
            all_verts,
            all_faces,
            K=camintr,
            rot=rot,
            trans=trans,
            image_sizes=[(width, height)],
            mode="rgb",
            faces_per_pixel=faces_per_pixel,
        )
        scene_res = {
            "body_info": body_info,
            "obj_infos": obj_infos,
            "scene_rend": rendres,
        }
        if viz_views:
            with torch.no_grad():
                viz_verts = all_verts.clone()
                viz_verts[:, :, 2] = -viz_verts[:, :, 2]
                viz_rendres = batch_render(
                    all_verts.new([0, 0, 0.7]) + viz_verts,
                    all_faces,
                    K=camintr,
                    rot=rot,
                    trans=trans,
                    image_sizes=[(width, height)],
                    mode="rgb",
                    faces_per_pixel=faces_per_pixel,
                )
            scene_res["scene_viz_rend"] = [viz_rendres]
        return scene_res

    def __len__(self):
        return len(self.data_df)

    def __repr__(self):
        return f"egoscene: {len(self.objects)} objects, {len(self)} scenes"
