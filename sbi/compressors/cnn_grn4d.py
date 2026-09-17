import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/ZhengyuLiang24/Conv4d-PyTorch/blob/main/Conv4d.py 
# https://medium.com/thedeephub/papers-explained-94-convnext-v2-2ecdabf2081c

class Conv4d(nn.Module):
    """
    Input:  [B, Cin, Z, D, H, W]
    Output: [B, Cout, Zout, Dout, Hout, Wout]
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)

        self.kz, self.kd, self.kh, self.kw = kernel_size
        self.sz, self.sd, self.sh, self.sw = stride
        self.pz, self.pd, self.ph, self.pw = padding

        self.spatial_convs = nn.ModuleList([
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(self.kd, self.kh, self.kw),
                stride=(self.sd, self.sh, self.sw),
                padding=(self.pd, self.ph, self.pw),
                bias=False
            )
            for _ in range(self.kz)
        ])
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        B, C, Z, D, H, W = x.shape
        x = F.pad(x, (0, 0, 0, 0, 0, 0, self.pz, self.pz))

        Z_padded = Z + 2 * self.pz
        Z_out = (Z_padded - self.kz) // self.sz + 1
        outputs = []

        for z_out in range(Z_out):
            z_start = z_out * self.sz
            out_z = None
            for k in range(self.kz):
                x_slice = x[:, :, z_start + k, :, :, :]
                y = self.spatial_convs[k](x_slice)

                if out_z is None:
                    out_z = y
                else:
                    out_z = out_z + y

            out_z = out_z + self.bias.view(1, -1, 1, 1, 1)
            outputs.append(out_z.unsqueeze(2))

        return torch.cat(outputs, dim=2)


class GroupNorm4d(nn.Module):
    """
    Standard GroupNorm fails on 6D tensors [B, C, Z, D, H, W].
    This reshaping layer temporarily collapses Z and D into the batch dim.
    """
    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        B, C, Z, D, H, W = x.shape
        # Collapse spatial depth frames into the batch context
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(B * Z * D, C, H, W)
        x = self.gn(x)
        # Restore original tensor dimensionality 
        x = x.view(B, Z, D, C, H, W).permute(0, 3, 1, 2, 4, 5).contiguous()
        return x


class GRN4d(nn.Module):
    """Global Response Normalization over 4D structures."""
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1, 1, 1))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3, 4, 5), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return x * (self.gamma * nx + 1) + self.beta


# ── Drop-In 4D Compressor ───────────────────────────────────────────────────
#Conv4DCompressor
class ResNet3DCompressor(nn.Module):
    def __init__(self, t_dim=8, n_params=4, direct=False):
        super().__init__()
        self.direct = bool(direct)
        
        if self.direct and t_dim != n_params:
            raise ValueError(f"direct regression needs t_dim==n_params, got t_dim={t_dim}, n_params={n_params}")

        # Assuming input structure contains a single channel [B, 1, Z, D, H, W]
        # Modify in_channels=3 if your raw cubes use 3-channel input paths.
        #self.conv1 = Conv4d(in_channels=3, out_channels=32, kernel_size=3, stride=(1, 2, 2, 2), padding=1)
        self.conv1 = Conv4d(in_channels=1, out_channels=32, kernel_size=3, stride=(1, 2, 2, 2), padding=1)
        self.gn1 = GroupNorm4d(4, 32)
        self.grn1 = GRN4d(32)

        self.conv2 = Conv4d(32, 64, kernel_size=3, stride=(1, 2, 2, 2), padding=1)
        self.gn2 = GroupNorm4d(8, 64)
        self.grn2 = GRN4d(64)

        self.conv3 = Conv4d(64, 128, kernel_size=3, stride=(1, 2, 2, 2), padding=1)
        self.gn3 = GroupNorm4d(8, 128)
        self.grn3 = GRN4d(128)

        self.conv4 = Conv4d(128, 256, kernel_size=3, stride=(1, 2, 2, 2), padding=1)
        self.gn4 = GroupNorm4d(8, 256)
        self.grn4 = GRN4d(256)

        self.conv5 = Conv4d(256, 512, kernel_size=(3, 2, 2, 2), stride=1, padding=0)
        self.gn5 = GroupNorm4d(8, 512)

        # ── Multi-Head Architectures identical to original baseline ─────────
        self.fc_summary = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(128, t_dim)
        )
        
        self.fc_aux = None if self.direct else nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(128, n_params)
        )

    def forward(self, x):
        f = self.features(x)
        t = self.fc_summary(f)
        if self.direct:
            return t, t
        return t, self.fc_aux(f)

    def features(self, x):
        x = self.grn1(F.leaky_relu(self.gn1(self.conv1(x))))
        x = self.grn2(F.leaky_relu(self.gn2(self.conv2(x))))
        x = self.grn3(F.leaky_relu(self.gn3(self.conv3(x))))
        x = self.grn4(F.leaky_relu(self.gn4(self.conv4(x))))
        x = F.leaky_relu(self.gn5(self.conv5(x)))
        return torch.flatten(x, 1)