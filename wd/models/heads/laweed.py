import torch
from torch.nn import functional as F

from .lawin import LawinAttn, ConvModule, PatchEmbed, LawinHead

LOW_LEVEL_PATCH_SIZE = 8
LOW_LEVEL_RATIO = 16


class LaweedHead(LawinHead):
    def __init__(self, in_channels: list, embed_dim=512, num_classes=19) -> None:
        super().__init__(in_channels, embed_dim, num_classes)

        setattr(self, f"lawin_low_{LOW_LEVEL_RATIO}",
                LawinAttn(embed_dim // 8, 16, patch_size=LOW_LEVEL_PATCH_SIZE))
        setattr(self, f"ds_low_{LOW_LEVEL_RATIO}", PatchEmbed(LOW_LEVEL_RATIO, embed_dim // 8, embed_dim // 8))
        self.low_cat = ConvModule(in_channels[0]*2, in_channels[0])

    def forward(self, features):
        B, _, H, W = features[1].shape
        outs = [self.linear_c2(features[1]).permute(0, 2, 1).reshape(B, -1, *features[1].shape[-2:])]

        for i, feature in enumerate(features[2:]):
            cf = eval(f"self.linear_c{i+3}")(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))

        feat = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        B, _, H, W = feat.shape

        ## Lawin attention spatial pyramid pooling
        feat_short = self.short_path(feat)
        feat_pool = F.interpolate(self.image_pool(feat), size=(H, W), mode='bilinear', align_corners=False)
        feat_lawin = self.get_lawin_att_feats(feat, 8, self.ratios)
        output = self.cat(torch.cat([feat_short, feat_pool, *feat_lawin], dim=1))

        ## Low-level feature enhancement
        low_attn = self.get_lawin_att_feats(features[0], LOW_LEVEL_PATCH_SIZE, [LOW_LEVEL_RATIO], step="low_")[0]
        low_cat = self.low_cat(torch.cat([low_attn, features[0]], dim=1))
        c1 = self.linear_c1(low_cat).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])
        output = F.interpolate(output, size=features[0].shape[-2:], mode='bilinear', align_corners=False)
        fused = self.low_level_fuse(torch.cat([output, c1], dim=1))

        seg = self.linear_pred(self.dropout(fused))
        return seg
