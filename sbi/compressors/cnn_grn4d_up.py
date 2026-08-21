import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv4d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 4
        if isinstance(stride, int):
            stride = (stride,) * 4
        if isinstance(padding, int):
            padding = (padding,) * 4

        self.kz, self.kd, self.kh, self.kw = kernel_size
        self.sz, self.sd, self.sh, self.sw = stride
        self.pz, self.pd, self.ph, self.pw = padding

        self.spatial_convs = nn.ModuleList(
            [
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=(self.kd, self.kh, self.kw),
                    stride=(self.sd, self.sh, self.sw),
                    padding=(self.pd, self.ph, self.pw),
                    bias=False,
                )
                for _ in range(self.kz)
            ]
        )
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
                out_z = y if out_z is None else out_z + y

            out_z = out_z + self.bias.view(1, -1, 1, 1, 1)
            outputs.append(out_z.unsqueeze(2))
        return torch.cat(outputs, dim=2)


class GroupNorm4d(nn.Module):
    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        B, C, Z, D, H, W = x.shape
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(B * Z * D, C, H, W)
        x = self.gn(x)
        return x.view(B, Z, D, C, H, W).permute(0, 3, 1, 2, 4, 5).contiguous()


class GRN4d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1, 1, 1))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3, 4, 5), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return x * (self.gamma * nx + 1) + self.beta


class ResidualBlock4d(nn.Module):
    """A clean 4D residual block using GroupNorm and GRN scaling."""

    def __init__(self, ci, co, stride=1):
        super().__init__()
        # Map tuple-based multi-spatial stride dimensions dynamically
        s_param = (1, stride, stride, stride) if isinstance(stride, int) else stride

        self.conv1 = Conv4d(ci, co, kernel_size=3, stride=s_param, padding=1)
        self.gn1 = GroupNorm4d(min(8, co // 4), co)
        self.grn = GRN4d(co)
        self.conv2 = Conv4d(co, co, kernel_size=3, stride=1, padding=1)
        self.gn2 = GroupNorm4d(min(8, co // 4), co)

        self.shortcut = nn.Sequential()
        if (
            ci != co
            or (isinstance(stride, int) and stride != 1)
            or (isinstance(stride, tuple) and any(s > 1 for s in stride))
        ):
            self.shortcut = nn.Sequential(
                Conv4d(ci, co, kernel_size=1, stride=s_param, padding=0),
                GroupNorm4d(min(8, co // 4), co),
            )

    def forward(self, x):
        out = self.grn(F.leaky_relu(self.gn1(self.conv1(x)), 0.1))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        return F.leaky_relu(out, 0.1)


class Conv4DCompressor(nn.Module):
    def __init__(self, t_dim=8, n_params=4, direct=False):
        super().__init__()
        self.direct = bool(direct)
        if self.direct and t_dim != n_params:
            raise ValueError(
                f"direct regression needs t_dim==n_params, got t_dim={t_dim}, n_params={n_params}"
            )

        # Stem handling 1 channel input [B, C=1, Z=3, D=32, H=32, W=32]
        self.stem = nn.Sequential(
            Conv4d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            GroupNorm4d(4, 32),
            nn.LeakyReLU(0.1),
        )

        # SUGGESTION 3: Set stride=1 on the early blocks to retain small-scale geometry
        self.layer1 = ResidualBlock4d(32, 32, stride=1)
        self.layer2 = ResidualBlock4d(32, 64, stride=2)
        self.layer3 = ResidualBlock4d(64, 128, stride=2)
        self.layer4 = ResidualBlock4d(128, 256, stride=2)
        self.layer5 = ResidualBlock4d(256, 512, stride=2)

        # Multi-scale feature concatenation space = 64 + 128 + 256 + 512 = 960
        combined_features_dim = 960

        # SUGGESTION 4: Summary branch dropout dropped to stabilize VMIM heads
        self.fc_summary = nn.Sequential(
            nn.Linear(combined_features_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            nn.Linear(128, t_dim),
        )

        self.fc_aux = (
            None
            if self.direct
            else nn.Sequential(
                nn.Linear(combined_features_dim, 256),
                nn.LayerNorm(256),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2),
                nn.Linear(128, n_params),
            )
        )

    def forward(self, x):
        x = self.stem(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        # 4D global pooling across Z, D, H, W (dims 2, 3, 4, 5)
        pool2 = torch.mean(x2, dim=(2, 3, 4, 5))
        pool3 = torch.mean(x3, dim=(2, 3, 4, 5))
        pool4 = torch.mean(x4, dim=(2, 3, 4, 5))
        pool5 = torch.mean(x5, dim=(2, 3, 4, 5))

        # SUGGESTION 3: Concatenation ensures micro-textures have a clean gradients bypass path
        f = torch.cat([pool2, pool3, pool4, pool5], dim=1)

        t = self.fc_summary(f)
        if self.direct:
            return t, t
        return t, self.fc_aux(f)
