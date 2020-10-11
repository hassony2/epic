import torch


class PerspectiveCamera(torch.nn.Module):
    def __init__(
        self, camintr, rot=None, trans=None, image_size=None, batch_size=1
    ):
        super().__init__()
        self.register_buffer("camintr", camintr)
        if rot is None:
            rot = torch.eye(3)
        self.register_buffer("rot", rot)
        if trans is None:
            trans = torch.zeros(3)
        self.register_buffer("trans", trans)
        self.camextr = torch.eye(4).to(rot.device)
        self.camextr[:3, :3] = rot

        self.camextr[:3, 3] = trans
        self.image_size = image_size

    def project(self, points):
        batch_size = points.shape[0]
        camextr = self.camextr.unsqueeze(0).repeat(batch_size, 1, 1)
        camintr = self.camintr.unsqueeze(0).repeat(batch_size, 1, 1)
        import pdb

        pdb.set_trace()
        return camintr, camextr

    def tocam3d(self, points):
        import pdb

        pdb.set_trace()
