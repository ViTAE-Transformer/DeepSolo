import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim])
                                    )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_point_pos_embed(pts_tensor, d_model, temp):
    # pts_tensor: bs, nq, n_pts, 2
    scale = 2 * math.pi
    dim = d_model // 2
    dim_t = torch.arange(dim, dtype=torch.float32, device=pts_tensor.device)
    dim_t = temp ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / dim)
    x_embed = pts_tensor[:, :, :, 0] * scale
    y_embed = pts_tensor[:, :, :, 1] * scale
    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_x, pos_y), dim=-1)
    return pos