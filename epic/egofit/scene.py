from epic.egofit.egohuman import EgoHuman
from epic.egofit.manobj import ManipulatedObject
import numpy as np


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
                obj_path, bboxes=bboxes, camintr=camera.get_camintr()
            )
            for obj_path in all_obj_paths[0]
        ]
        self.camera = camera
        self.debug = debug

    def get_optim_params(self):
        # Collect EgoHuman parameters
        params = self.egohuman.get_params()

        # Aggregate object parameters
        for obj in self.objects:
            obj_params = obj.get_params()
            for obj_param in obj_params:
                params.append(obj_param)
        return params

    def forward(self):
        body_info = self.egohuman.forward()
        obj_infos = [obj.forward() for obj in self.objects]

        # Render scene
        import pdb

        pdb.set_trace()
        return {"body_info": body_info, "obj_infos": obj_infos}

    def __len__(self):
        return len(self.data_df)

    def __repr__(self):
        return f"egoscene: {len(self.objects)} objects, {len(self)} scenes"
