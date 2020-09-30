import torch


def get_axis(axes, row_idx=0, col_idx=0, row_nb=1, col_nb=1):
    if row_nb == 1 and col_nb == 1:
        ax = axes
    elif row_nb == 1:
        ax = axes[col_idx]
    elif col_nb == 1:
        ax = axes[row_idx]
    else:
        ax = axes[row_idx, col_idx]
    return ax


def numpify(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    return tensor
