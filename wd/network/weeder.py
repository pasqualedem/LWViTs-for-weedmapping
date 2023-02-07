from functools import reduce
import torch
import math
import torch.nn.functional as F
import torch.nn as nn

from cc_torch import connected_components_labeling
from einops import rearrange


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiheadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super().__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )


def parse_pos(pos, size):
    return pos // size, pos % size


def get_patch(x, pos, size):
    if pos == -1:
        return torch.zeros((x.shape[0], size, size), device=x.device)
    cx, cy = parse_pos(pos, x.shape[1])
    return F.pad(x[:, cx:cx + size, cy:cy + size], 
                 (0, max(0, size - (x.shape[2] - cy)), 0, max(0, size - (x.shape[1] - cx))))


class PatchExtractor(nn.Module):
    def __init__(self, patch_dim):
        super().__init__()
        self.patch_dim = patch_dim
        self.patch_pool = nn.AdaptiveAvgPool2d((patch_dim, patch_dim))
    
    def forward(self, x, pos, mask):
        if pos == -1:
            return torch.zeros((x.shape[0], self.patch_dim, self.patch_dim), device=x.device)
        xs, ys = torch.where(mask == pos)
        x0 = xs.min()
        x1 = xs.max() + 1
        y0 = ys.min()
        y1 = ys.max() + 1
        patch = x[:, x0:x1, y0:y1]
        return self.patch_pool(patch)


class WeedLayer(torch.nn.Module):
    def __init__(self, in_channels, embedding_dim, patch_dim, emb_patch_div, num_heads, num_classes, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.patch_dim = patch_dim
        self.attention = WeedAttention(in_channels, embedding_dim, patch_dim, emb_patch_div, num_heads)
        self.context_pooling = torch.nn.AvgPool2d(
            kernel_size=(3, 3), stride=2, padding=1
        )
        self.classifier = torch.nn.Linear(embedding_dim, num_classes - 1) # remove background
        self.patch_extractor = PatchExtractor(patch_dim)
    
    def _extract_patches(self, features, probs):
        samples = probs.argmax(dim=1).type(torch.uint8)
        masks = ([
            connected_components_labeling(sample)
            for sample in samples
        ])
        
        # positions = [torch.unique(mask) - 1 for mask in masks] # shifted by 1 cause 0 is background
        # positions = [pos[pos != -1] for pos in positions] # remove background
        positions = [torch.unique(mask) for mask in masks]
        positions = [pos[pos != 0] for pos in positions]
        
        filled_positions = [torch.tensor([-1]) if len(pos) == 0 else pos for pos in positions] # if no plant, add a dummy position
        
        patches = ([torch.stack([self.patch_extractor(feature, pos, mask) for pos in position], dim=0) 
            for feature, position, mask in zip(features, filled_positions, masks)
        ])
        patches = [rearrange(patch, 'p c h w -> p (c h w)') for patch in patches]

        # probs_patches = ([torch.stack([get_patch(feature, pos, self.patch_dim) for pos in position], dim=0) 
        #     for feature, position in zip(probs, positions)
        # ])
        # probs_patches = [rearrange(patch, 'p c h w -> p (c h w)') for patch in probs_patches]
        return patches, masks, positions, filled_positions
    
    def _fill_patch(self, patch, fill_shape):
        return F.pad(patch, (0, 0, 0, fill_shape))
    
    def _fill_patches(self, patches, max_len):
        return torch.stack([self._fill_patch(patch, max_len - patch.shape[0]) for patch in patches], dim=0)
    
    def _reclassify(self, probs, patch_logits, mask, positions):
        if len(positions) == 0: # no plant detected -> positions empty with dummy patch
            return probs
        patch_logits = patch_logits[:positions.shape[0]]
        for patch_logit, position in zip(patch_logits, positions):
            mask_position = mask == position
            orig_mean_logits = rearrange(probs[mask_position.repeat(3, 1, 1)], "(cl v) -> cl v", cl=3).mean(dim=1) + 1e-6
            
            probs[0, ::][mask_position] = (orig_mean_logits[0] / orig_mean_logits[1:3].mean()) * patch_logit.mean()
            probs[1, ::][mask_position] = patch_logit[0] # weed
            probs[2, ::][mask_position] = patch_logit[1] # crop
        return probs

    def forward(self, features, probs):
        patches, masks, positions, filled_positions = self._extract_patches(features, probs)
        if reduce(lambda x, y: x + y, [len(position) for position in positions]) == 0:
            return probs
        
        shapes = torch.tensor([len(position) for position in filled_positions])

        max_len = shapes.max()

        patches = self._fill_patches(patches, max_len) # [B, max_len, C * H * W]
        # probs_patches = self._fill_patches(probs_patches, max_len) # [B, max_len, C * H * W]
        # patch_mask = torch.zeros(shapes.shape[0], max_len, dtype=torch.bool)

        # for i, shape in enumerate(shapes):
        #     patch_mask[i, 0:shape] = True

        patches = rearrange(patches, 'b p (c h w) -> b p c h w', h=self.patch_dim, w=self.patch_dim)
        # probs_patches = rearrange(probs_patches, 'b p (c h w) -> b p c h w', h=self.patch_dim, w=self.patch_dim)
        
        attn = self.attention(features, patches)
        outs = self.classifier(attn)
        
        for i in range(len(positions)):
            probs[i] = self._reclassify(probs[i], outs[i], masks[i], positions[i])
        return probs    
        
        
class Patch2vec(nn.Module):
    def __init__(self, in_channels, emb_features, patch_dim, emb_patch_div, num_heads):
        super().__init__()
        self.patch_dim = patch_dim
        self.num_heads = num_heads
        self.in_features = in_channels
        
        result_size = patch_dim // emb_patch_div
        channel_depth = in_channels // 4
        
        self.patch_pooling = nn.AdaptiveAvgPool3d((channel_depth, result_size, result_size))
        fvector_size = channel_depth * result_size * result_size
        self.linear_reducer = nn.Linear(fvector_size, emb_features)
        
    def forward(self, patches):
        B, P, C, H, W = patches.shape
        patches = rearrange(patches, "b p c h w -> (b p) c h w")
        patches = rearrange(self.patch_pooling(patches), "(b p) c h w -> b p (c h w)", p=P)
        return self.linear_reducer(patches) #  [B, P, emb_features]


class WeedAttention(nn.Module):
    def __init__(self, in_channels, emb_features, patch_dim, emb_patch_div, num_heads, p_drop=0.):
        super().__init__()
        self.patch_dim = patch_dim
        self.num_heads = num_heads
        self.in_features = in_channels
        
        self.patch_embedder = Patch2vec(in_channels, emb_features, patch_dim, emb_patch_div, num_heads)
        
        self.attention = nn.MultiheadAttention(emb_features, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(emb_features)
        self.mlp = nn.Sequential(nn.Linear(emb_features, emb_features), nn.GELU(), nn.Dropout(p_drop))

    def forward(self, features, patches, key_padding_mask=None):
        """
        features: [B, C, H, W]
        patches: [B, P, C, H, W]
        probs: [B, P, CLASSES, Hp, Wp]
        """
        P = patches.shape[1]
        B, C, H, W = features.shape
        patches = self.norm(self.patch_embedder(patches))
        
        blocks = rearrange(F.unfold(features, kernel_size=(self.patch_dim, self.patch_dim), stride=16), 
                           'b (c h w) l -> b l c h w', h=self.patch_dim, w=self.patch_dim)
        blocks = self.norm(self.patch_embedder(blocks))
        
        attn, _ = self.attention(patches, blocks, blocks)
        res = self.norm(attn + patches)
        return self.mlp(res) + res
