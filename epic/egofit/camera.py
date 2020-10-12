import torch
from epic.lib3d import ops3d


class PerspectiveCamera(torch.nn.Module):
    def __init__(
        self,
        camintr,
        rot=None,
        trans=None,
        image_size=None,
        batch_size=1,
        min_z=0.1,
    ):
        super().__init__()
        self.min_z = min_z
        self.register_buffer("camintr", camintr)
        if rot is None:
            rot = torch.eye(3)
        self.register_buffer("rot", rot)
        if trans is None:
            trans = torch.zeros(3)
        self.register_buffer("trans", trans)
        camextr = torch.eye(4).to(rot.device)
        camextr[:3, :3] = rot

        camextr[:3, 3] = trans
        self.register_buffer("camextr", camextr)

        self.image_size = image_size

    def project(self, points):
        batch_size = points.shape[0]
        camextr = self.camextr.unsqueeze(0).repeat(batch_size, 1, 1)
        camintr = self.camintr.unsqueeze(0).repeat(batch_size, 1, 1)
        points2d = ops3d.project(points, camintr, camextr, min_z=self.min_z)
        return points2d

    def tocam3d(self, points):
        import pdb

        pdb.set_trace()
