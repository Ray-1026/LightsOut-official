import torch
from torch import nn
from torch.nn import init
from torchvision.models import resnet34, resnet50
import torchvision.models.vision_transformer as vit


class LightRegressModel(nn.Module):
    def __init__(self, num_lights=4, alpha=2.0, beta=8.0):
        super(LightRegressModel, self).__init__()

        self.num_lights = num_lights
        self.alpha = alpha
        self.beta = beta

        self.model = resnet34(pretrained=True)
        # self.model = resnet50(pretrained=True)
        self.init_resnet()

        # self.model = vit.vit_b_16(pretrained=True)
        # self.init_vit()

        self.xyr_mlp = nn.Sequential(
            nn.Linear(self.last_dim, 3 * self.num_lights),
        )
        self.p_mlp = nn.Sequential(
            nn.Linear(self.last_dim, self.num_lights),
            nn.Sigmoid(),
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

            # 設定新的 positional embedding
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

    # def render(self, canvas_width, canvas_height, shapes, shape_groups, samples=2):
    #     _render = pydiffvg.RenderFunction.apply
    #     scene_args = pydiffvg.RenderFunction.serialize_scene(
    #         canvas_width, canvas_height, shapes, shape_groups
    #     )
    #     img = _render(
    #         canvas_width, canvas_height, samples, samples, 0, None, *scene_args
    #     )
    #     return img

    def forward(self, x, height=512, width=512, smoothness=0.1, merge=False):
        _x = self.model(x)  # [B, last_dim]

        _xyr = self.xyr_mlp(_x)
        _xyr = _xyr.view(-1, self.num_lights, 3)

        _p = self.p_mlp(_x)
        _p = _p.view(-1, self.num_lights)

        output = torch.cat([_xyr, _p.unsqueeze(-1)], dim=-1)

        return output

    # def forward_render(self, x, height=512, width=512):
    #     _x = self.forward(x)

    #     _xy = _x[:, :, :2]
    #     _r = _x[:, :, 2]
    #     _p = _x[:, :, 3]

    #     masks = None
    #     for b in range(_x.size(0)):
    #         # xy, r = _x[b, :, :2], _x[b, :, 2]

    #         shapes, shape_groups = [], []
    #         n = 0
    #         for i in range(self.num_lights):
    #             if _r[b, i] < 0 or _r[b, i] > 1 or _p[b, i] < 0.5:
    #                 continue

    #             # diffvg
    #             shapes += [
    #                 pydiffvg.Circle(radius=_r[b, i] * height, center=_xy[b, i] * width)
    #             ]
    #             # print(shapes[-1].radius, shapes[-1].center)
    #             shape_groups += [
    #                 pydiffvg.ShapeGroup(
    #                     shape_ids=torch.tensor([n]),
    #                     fill_color=torch.tensor(
    #                         [1.0, 1.0, 1.0, 1.0], requires_grad=False
    #                     ),
    #                 )
    #             ]
    #             n += 1

    #         if len(shapes) == 0:
    #             img = torch.zeros(height, width, 4, device=x.device)
    #         else:
    #             img = self.render(width, height, shapes, shape_groups, samples=2)

    #         img = img.permute(2, 0, 1).view(4, height, width)[:3].mean(0, keepdim=True)
    #         img = img.unsqueeze(0)  # [1, 1, H, W]

    #         masks = (
    #             img if masks is None else torch.cat([masks, img], dim=0)
    #         )  # [B, 1, H, W]

    #     return masks  # [B, 1, H, W]

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
    model = LightRegressModel(num_lights=4).cuda()
    x = torch.randn(8, 3, 512, 512, device="cuda")
    y = model.forward_render(x)
    print(y.shape)
