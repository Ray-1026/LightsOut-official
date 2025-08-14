import torch
from torch import nn
from torch.nn import init
from torchvision.models import resnet34, resnet50
import torchvision.models.vision_transformer as vit


class LightSourceRegressor(nn.Module):
    def __init__(self, num_lights=4, alpha=2.0, beta=8.0, **kwargs):
        super(LightSourceRegressor, self).__init__()

        self.num_lights = num_lights
        self.alpha = alpha
        self.beta = beta

        self.model = resnet34(pretrained=True)
        # self.model = resnet50(pretrained=True)
        # self.model = vit.vit_b_16(pretrained=True)
        self.init_resnet()
        # self.init_vit()

        self.xyr_mlp = nn.Sequential(
            nn.Linear(self.last_dim, 3 * self.num_lights),
        )
        self.p_mlp = nn.Sequential(
            nn.Linear(self.last_dim, self.num_lights),
            nn.Sigmoid(),  # ensure p is in [0, 1]
        )

    def init_resnet(self):
        self.last_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()

    def init_vit(self):
        self.model.image_size = 512
        old_pos_embed = self.model.encoder.pos_embedding
        num_patches_old = (224 // 16) ** 2
        num_patches_new = (512 // 16) ** 2

        if num_patches_new != num_patches_old:
            old_pos_embed = old_pos_embed[:, 1:]
            old_pos_embed = nn.functional.interpolate(
                old_pos_embed.permute(0, 2, 1), size=(num_patches_new,), mode="linear"
            )
            old_pos_embed = old_pos_embed.permute(0, 2, 1)

            # new positional embedding
            self.model.encoder.pos_embedding = nn.Parameter(
                torch.cat(
                    [self.model.encoder.pos_embedding[:, :1], old_pos_embed], dim=1
                )
            )

        # num_classes = 4 * self.num_lights  # x, y, r, p
        # self.model.heads.head = nn.Linear(self.model.hidden_dim, num_classes)

        # remove the head
        self.last_dim = self.model.hidden_dim
        self.model.heads.head = nn.Identity()

    def forward(self, x, height=512, width=512, smoothness=0.1, merge=False):
        _x = self.model(x)  # [B, last_dim]

        _xyr = self.xyr_mlp(_x)
        _xyr = _xyr.view(-1, self.num_lights, 3)

        _p = self.p_mlp(_x)
        _p = _p.view(-1, self.num_lights)

        output = torch.cat([_xyr, _p.unsqueeze(-1)], dim=-1)

        return output

    def forward_render(self, x, height=512, width=512, smoothness=0.1, merge=False):
        _x = self.forward(x)

        _xy = _x[:, :, :2]
        _r = _x[:, :, 2]
        _p = _x[:, :, 3]

        masks = None
        masks_merge = None
        for b in range(_x.size(0)):
            x, y, r = _xy[b, :, 0] * width, _xy[b, :, 1] * width, _r[b] * width / 2
            p = _p[b]

            mask_list = []
            for i in range(self.num_lights):
                if r[i] < 0 or r[i] > width or p[i] < 0.5:
                    continue

                y_coords, x_coords = torch.meshgrid(
                    torch.arange(height, device=x.device),
                    torch.arange(width, device=x.device),
                    indexing="ij",
                )

                distances = torch.sqrt((x_coords - x[i]) ** 2 + (y_coords - y[i]) ** 2)
                mask_i = torch.sigmoid(smoothness * (r[i] - distances))
                mask_list.append(mask_i)

            if len(mask_list) == 0:
                _mask_merge = torch.zeros(1, 1, height, width, device=x.device)
            else:
                _mask_merge = torch.stack(mask_list, dim=0).sum(dim=0).unsqueeze(0)
                _mask_merge = _mask_merge.unsqueeze(0)

            masks_merge = (
                _mask_merge
                if masks_merge is None
                else torch.cat([masks_merge, _mask_merge], dim=0)
            )

        masks_merge = torch.clamp(masks_merge, 0, 1)

        return masks_merge  # [B, 1, H, W]


if __name__ == "__main__":
    # pydiffvg.set_use_gpu(torch.cuda.is_available())
    model = LightSourceRegressor(num_lights=4).cuda()
    x = torch.randn(8, 3, 512, 512, device="cuda")
    y = model.forward_render(x)
    print(y.shape)
